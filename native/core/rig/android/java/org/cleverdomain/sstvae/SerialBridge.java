package org.cleverdomain.sstvae;

import android.Manifest;
import android.app.Activity;
import android.app.PendingIntent;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothManager;
import android.bluetooth.BluetoothSocket;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.hardware.usb.UsbDevice;
import android.hardware.usb.UsbDeviceConnection;
import android.hardware.usb.UsbInterface;
import android.hardware.usb.UsbManager;
import android.os.Build;
import android.util.Log;

import com.hoho.android.usbserial.driver.Cp21xxSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialDriver;
import com.hoho.android.usbserial.driver.UsbSerialPort;
import com.hoho.android.usbserial.driver.UsbSerialProber;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The Java half of {@code core/rig/android/}: a serial link to a radio, over
 * USB or Bluetooth.
 *
 * <p><b>Direction rule, and it is the opposite of {@link AudioBridge}'s.</b>
 * Audio is a stream, so Java owns a thread and pushes into C++. A rig link is
 * request and response, driven by {@code LoopbackBridge}'s pump threads, so
 * C++ calls in here. The one exception is the USB permission result, which
 * arrives on a broadcast receiver and goes out through {@code
 * nativePermissionResult}.
 *
 * <p><b>Device identifiers are vendor and product, never the device node.</b>
 * {@code UsbDevice.getDeviceName()} is {@code /dev/bus/usb/001/007} and changes
 * on every replug, so a setting keyed on it stops working the first time the
 * operator unplugs the radio. {@code usb:10c4:ea60} survives.
 *
 * <p><b>But a vendor and product pair names a *kind* of device, and a radio can
 * present two of the same kind.</b> This paragraph used to end "two identical
 * adapters are indistinguishable and the first wins — the same accepted
 * ambiguity the audio layer takes on device names", and that was wrong in a way
 * that cost days: an IC-9700 exposes its CI-V port and its USB serial function
 * as two separate USB devices sharing {@code 10c4:ea60}, so the id named both,
 * the picker showed two identical rows, and the lookup returned whichever
 * {@code getDeviceList()} — a {@code HashMap} — happened to yield first. The
 * app talked to a healthy chip that is not wired to the radio's CI-V engine,
 * which from the outside is indistinguishable from a radio that ignores it.
 *
 * <p>So an id carries two independent suffixes, and they are <em>not</em> the
 * same axis: {@code #p} is the p'th port of one multi-port driver (a CP2105),
 * and {@code @u} is the u'th device sharing a vendor and product pair. Both are
 * omitted at zero, so an id saved before they existed still means the first
 * port of the first device. Units are ordered by {@code getDeviceName()},
 * which is the only ordering available before permission is granted and is not
 * promised across a replug — hence the row label says which is which and the
 * operator chooses, because nothing readable tells an Icom's CI-V port from
 * its data port.
 *
 * <p>Call {@link #init} once with an application context before anything else.
 */
public final class SerialBridge {

    private static final String TAG = "sstvae";

    /**
     * Our own permission-result action.
     *
     * <p>Package-qualified rather than something generic because it is
     * broadcast to ourselves and nothing else has any business seeing it.
     */
    private static final String ACTION_GRANT_USB = "org.cleverdomain.sstvae.USB_PERMISSION";

    /** The Serial Port Profile. The one UUID every RFCOMM radio answers on. */
    private static final UUID SPP = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");

    private static final int REQUEST_BLUETOOTH_CONNECT = 1002;

    /** How long a CAT command may take to reach the radio before it is a fault. */
    private static final int WRITE_TIMEOUT_MS = 2000;

    private static Context context;
    private static BroadcastReceiver receiver;
    private static final AtomicInteger nextToken = new AtomicInteger(1);
    private static final Map<Integer, Link> links = new ConcurrentHashMap<>();

    private SerialBridge() {}

    public static void init(Context appContext) {
        context = appContext.getApplicationContext();
        registerReceiver();
    }

    // --- permission -------------------------------------------------------

    private static void registerReceiver() {
        if (receiver != null) return;
        receiver = new BroadcastReceiver() {
            @Override
            public void onReceive(Context c, Intent intent) {
                if (!ACTION_GRANT_USB.equals(intent.getAction())) return;
                final boolean granted =
                        intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false);
                final UsbDevice device = usbExtra(intent);
                nativePermissionResult(
                        device == null ? ""
                                : usbId(device, usbUnitOf(usbManager(), device), 0),
                        granted);
            }
        };
        final IntentFilter filter = new IntentFilter(ACTION_GRANT_USB);
        if (Build.VERSION.SDK_INT >= 33) {
            // Required from API 33 up, and the app targets 36. The
            // broadcast is ours to ourselves, so NOT_EXPORTED is both the
            // safe answer and the correct one.
            context.registerReceiver(receiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            context.registerReceiver(receiver, filter);
        }
    }

    /**
     * {@code EXTRA_DEVICE}, without the API 33 deprecation warning.
     *
     * <p>The untyped {@code getParcelableExtra} still works on 33+ and is what
     * every example uses, but the app targets 36 and a deprecated call in a
     * file nobody looks at is how a removal gets discovered by a build break.
     */
    @SuppressWarnings("deprecation")
    private static UsbDevice usbExtra(Intent intent) {
        if (Build.VERSION.SDK_INT >= 33) {
            return intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice.class);
        }
        return intent.getParcelableExtra(UsbManager.EXTRA_DEVICE);
    }

    static boolean hasPermission(String id) {
        if (id == null) return false;
        if (id.startsWith("bt:")) return bluetoothPermitted();
        final UsbManager manager = usbManager();
        if (manager == null) return false;
        final UsbDevice device = findUsbDevice(id);
        return device != null && manager.hasPermission(device);
    }

    static void requestPermission(String id) {
        if (id == null) return;
        if (id.startsWith("bt:")) {
            requestBluetoothPermission();
            return;
        }
        final UsbManager manager = usbManager();
        final UsbDevice device = findUsbDevice(id);
        if (manager == null || device == null) {
            nativePermissionResult(id, false);
            return;
        }
        // **Three things here are Android 14 requirements, not style.**
        // A mutable PendingIntent may not carry an implicit intent, so
        // the intent names our own package; the USB framework has to be
        // able to fill in EXTRA_DEVICE, so it cannot be immutable
        // either. Getting this wrong throws at the moment the operator
        // taps the device, which is a long way from where it is written.
        final Intent intent = new Intent(ACTION_GRANT_USB).setPackage(context.getPackageName());
        final int flags = Build.VERSION.SDK_INT >= 31
                ? PendingIntent.FLAG_MUTABLE
                : 0;
        manager.requestPermission(device, PendingIntent.getBroadcast(context, 0, intent, flags));
    }

    private static boolean bluetoothPermitted() {
        // BLUETOOTH_CONNECT became a runtime permission at API 31. Below
        // that the manifest permission is enough and there is nothing to
        // ask for.
        if (Build.VERSION.SDK_INT < 31) return true;
        return context != null
                && context.checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT)
                        == PackageManager.PERMISSION_GRANTED;
    }

    private static void requestBluetoothPermission() {
        if (bluetoothPermitted()) {
            nativePermissionResult("bt:", true);
            return;
        }
        // Needs an Activity, and `context` is deliberately the
        // *application* context, so it has to be fetched. The result
        // comes back through onRequestPermissionsResult, which Qt owns;
        // rather than reach into that, the settings screen re-reads
        // `hasPermission` when it regains focus.
        final Activity activity = currentActivity();
        if (activity == null) {
            nativePermissionResult("bt:", false);
            return;
        }
        activity.requestPermissions(
                new String[] {Manifest.permission.BLUETOOTH_CONNECT},
                REQUEST_BLUETOOTH_CONNECT);
    }

    /**
     * Qt's activity, which is not the application context this class was
     * given. Reflection because Qt's bindings are not a compile-time
     * dependency of this layer, and adding one would tie the rig transport to
     * a toolkit it has nothing to do with.
     */
    private static Activity currentActivity() {
        try {
            final Class<?> qt = Class.forName("org.qtproject.qt.android.QtNative");
            final Object activity = qt.getMethod("activity").invoke(null);
            return activity instanceof Activity ? (Activity) activity : null;
        } catch (Exception e) {
            Log.w(TAG, "no Qt activity for a permission request: " + e);
            return null;
        }
    }

    // --- enumeration ------------------------------------------------------
    //
    // One string per device rather than parallel arrays: "id\tlabel\t0|1".
    // Three arrays could be returned out of step with each other and a device
    // would then be described by another device's row.

    static String[] usbDevices() {
        final UsbManager manager = usbManager();
        final List<String> out = new ArrayList<>();
        if (manager == null) return new String[0];
        // **Sorted, because `findAllDrivers` is not.** It walks
        // `getDeviceList().values()` -- a `HashMap` -- so without this
        // the rows come out in whatever order the map felt like, and
        // "(2 of 2)" can appear above "(1 of 2)". The unit *index* was
        // already sorted; this is the same `HashMap` leaking through a
        // second time, one layer up. Ordered by vendor, product, then
        // device name so that ports of one radio group together and
        // their order matches the numbering in their own labels.
        final List<UsbSerialDriver> drivers =
                new ArrayList<>(UsbSerialProber.getDefaultProber().findAllDrivers(manager));
        drivers.sort((a, b) -> {
            final UsbDevice da = a.getDevice();
            final UsbDevice db = b.getDevice();
            if (da.getVendorId() != db.getVendorId()) {
                return Integer.compare(da.getVendorId(), db.getVendorId());
            }
            if (da.getProductId() != db.getProductId()) {
                return Integer.compare(da.getProductId(), db.getProductId());
            }
            return deviceName(da).compareTo(deviceName(db));
        });

        for (UsbSerialDriver driver : drivers) {
            final UsbDevice device = driver.getDevice();
            final boolean permitted = manager.hasPermission(device);
            final int ports = driver.getPorts().size();
            // **Two things can multiply a row, and they are not the
            // same thing.** `ports` is one driver's several UARTs (a
            // CP2105); `units` is several devices that happen to share
            // a VID:PID, which is how an IC-9700 presents its CI-V port
            // and its USB serial function. Conflating them is what made
            // the picker show two identical rows that both addressed
            // the same chip.
            final int unit = usbUnitOf(manager, device);
            final int units =
                    usbUnits(manager, device.getVendorId(), device.getProductId()).size();
            for (int i = 0; i < ports; i++) {
                // **The marker goes in front of the name, not after
                // it.** Said plainly because the operator is the only
                // one who can tell these apart -- nothing readable
                // distinguishes an Icom's CI-V port from its data port,
                // so the answer is to try the other one. Put after the
                // name it is the first thing a narrow combo elides, and
                // eliding the only distinguishing text leaves two rows
                // that read identically: "Silicon Labs CP2102N USB to
                // UART Bridge Contr...". In front it survives both the
                // closed control and the open list.
                final StringBuilder marker = new StringBuilder();
                if (units > 1) marker.append(unit + 1).append(" of ").append(units);
                if (ports > 1) {
                    if (marker.length() > 0) marker.append(", ");
                    marker.append("port ").append(i + 1);
                }
                final String label = marker.length() > 0
                        ? "(" + marker + ") " + describe(device)
                        : describe(device);
                out.add(usbId(device, unit, i) + "\t" + clean(label) + "\t"
                        + (permitted ? "1" : "0"));
            }
        }
        return out.toArray(new String[0]);
    }

    static String[] bluetoothDevices() {
        if (!bluetoothPermitted()) return new String[0];
        final BluetoothAdapter adapter = bluetoothAdapter();
        if (adapter == null) return new String[0];
        final List<String> out = new ArrayList<>();
        try {
            // **Bonded devices only.** Discovery needs BLUETOOTH_SCAN and,
            // before API 31, location permission — a large ask for a list
            // whose whole content is the radio the operator already paired
            // in system Settings.
            final Collection<BluetoothDevice> bonded = adapter.getBondedDevices();
            if (bonded == null) return new String[0];
            // Sorted before the rows are built, for the same reason
            // the USB list is: `getBondedDevices()` returns a Set, so
            // without this the rows shuffle between refreshes for no
            // reason the operator can see. By what they are reading --
            // the name, falling back to the address when there is none.
            final List<BluetoothDevice> paired = new ArrayList<>(bonded);
            paired.sort((a, b) -> bluetoothLabel(a).compareToIgnoreCase(bluetoothLabel(b)));
            for (BluetoothDevice device : paired) {
                out.add("bt:" + device.getAddress() + "\t"
                        + clean(bluetoothLabel(device)) + "\t1");
            }
        } catch (SecurityException e) {
            // The permission was revoked between the check and the call.
            return new String[0];
        }
        return out.toArray(new String[0]);
    }

    /**
     * What was actually opened, for the rig trace.
     *
     * <p><b>Written because a failing radio and a mis-driven chip look
     * identical from above.</b> An Elecraft K4 works over this path and
     * two Icoms do not, and everything above the USB layer -- Hamlib's
     * trace, the bridge, the byte counts -- says exactly the same thing
     * in both cases. What it cannot say is which driver
     * {@code UsbSerialProber} picked, which USB interface that driver
     * claimed, or how many interfaces the device has to choose from,
     * and those are the parts that differ between an FTDI on its own
     * and a vendor bridge inside a composite device.
     *
     * <p>Never throws: this runs on the rig worker thread inside an
     * open that is already in progress, and a diagnostic that can fail
     * the operation it is diagnosing is worse than none.
     */
    static String describeLink(String id) {
        if (id == null) return "";
        try {
            if (id.startsWith("bt:")) return "bluetooth " + id.substring(3);
            if (!id.startsWith("usb:")) return id;
            final UsbManager manager = usbManager();
            final UsbDevice device = findUsbDevice(id);
            if (manager == null || device == null) return id + " (not present)";
            final StringBuilder out = new StringBuilder();
            out.append(String.format(Locale.US, "%04x:%04x", device.getVendorId(),
                    device.getProductId()));
            out.append(" \"").append(describe(device)).append('"');
            // Which of the same-VID:PID devices this is, and how many
            // there are. The fact that answers "are we even talking to
            // the CI-V port".
            final int units =
                    usbUnits(manager, device.getVendorId(), device.getProductId()).size();
            out.append(", unit ").append(usbUnitOf(manager, device) + 1)
               .append(" of ").append(units);
            final int interfaces = device.getInterfaceCount();
            out.append(", ").append(interfaces).append(" interface(s) [");
            for (int i = 0; i < interfaces; i++) {
                if (i != 0) out.append(' ');
                final UsbInterface iface = device.getInterface(i);
                out.append(i).append(":cls").append(iface.getInterfaceClass())
                   .append('/').append(iface.getEndpointCount()).append("ep");
            }
            out.append(']');
            final UsbSerialDriver driver =
                    UsbSerialProber.getDefaultProber().probeDevice(device);
            if (driver == null) {
                out.append(", no driver");
            } else {
                out.append(", driver ")
                   .append(driver.getClass().getSimpleName())
                   .append(", port ").append(usbPortIndex(id))
                   .append(" of ").append(driver.getPorts().size());
            }
            return out.toString();
        } catch (RuntimeException e) {
            return id + " (could not describe: " + e + ")";
        }
    }

    /** "Name (AA:BB:CC:DD:EE:FF)", or just the address when unnamed. */
    private static String bluetoothLabel(BluetoothDevice device) {
        String name = null;
        try {
            name = device.getName();
        } catch (SecurityException e) {
            // Revoked between the check and the call; the address still
            // identifies it.
        }
        return (name == null || name.isEmpty())
                ? device.getAddress()
                : name + " (" + device.getAddress() + ")";
    }

    private static String describe(UsbDevice device) {
        String product = null;
        String manufacturer = null;
        try {
            product = device.getProductName();
            manufacturer = device.getManufacturerName();
        } catch (SecurityException e) {
            // Some names need permission we do not have yet; the ids
            // below are always readable and are what identifies it.
        }
        if (product != null && !product.trim().isEmpty()) {
            if (manufacturer != null && !manufacturer.trim().isEmpty()) {
                return manufacturer.trim() + " " + product.trim();
            }
            return product.trim();
        }
        return String.format(Locale.US, "USB %04x:%04x",
                device.getVendorId(), device.getProductId());
    }

    /** Tabs are the field separator, so they cannot appear inside a field. */
    private static String clean(String s) {
        return s.replace('\t', ' ').replace('\n', ' ');
    }

    /**
     * Every attached device with this one's vendor and product ids, in a
     * stable order.
     *
     * <p><b>There can be more than one, and assuming otherwise cost an
     * Icom.</b> An IC-9700 presents its CI-V port and its USB serial
     * function as two separate USB devices that share a VID:PID — so
     * `usb:10c4:ea60` named both, the picker showed two identical rows,
     * and the lookup returned whichever one {@code getDeviceList()}
     * happened to yield first. Out of a {@code HashMap}, so not even the
     * same one twice. Bytes went out of a chip that was never wired to
     * the radio's CI-V engine, which is exactly what "the chip
     * transmitted and the radio said nothing" looks like from above.
     *
     * <p>Sorted by {@code getDeviceName()} — the {@code
     * /dev/bus/usb/BBB/DDD} path — because it is the only ordering
     * available before permission is granted, and a fixed cable on a
     * fixed hub enumerates in a fixed order. It is not promised across a
     * replug, which is why the *label* says which is which and the
     * operator picks; guessing which port is CI-V is not something this
     * layer can do.
     */
    private static List<UsbDevice> usbUnits(UsbManager manager, int vendor, int product) {
        final List<UsbDevice> units = new ArrayList<>();
        if (manager == null) return units;
        for (UsbDevice device : manager.getDeviceList().values()) {
            if (device.getVendorId() == vendor && device.getProductId() == product) {
                units.add(device);
            }
        }
        units.sort((a, b) -> deviceName(a).compareTo(deviceName(b)));
        return units;
    }

    /** `getDeviceName()`, never null, so it can be compared and sorted. */
    private static String deviceName(UsbDevice device) {
        final String name = device.getDeviceName();
        return name == null ? "" : name;
    }

    /** Where `device` sits in {@link #usbUnits}, or 0 if it is not there. */
    private static int usbUnitOf(UsbManager manager, UsbDevice device) {
        final List<UsbDevice> units =
                usbUnits(manager, device.getVendorId(), device.getProductId());
        for (int i = 0; i < units.size(); i++) {
            if (deviceName(units.get(i)).equals(deviceName(device))) return i;
        }
        return 0;
    }

    /**
     * `usb:VVVV:PPPP`, plus `@u` for the u'th device sharing those ids and
     * `#p` for the p'th port of a multi-port driver. Both suffixes are
     * omitted at zero, so an id saved before they existed still means
     * the first port of the first device.
     */
    private static String usbId(UsbDevice device, int unit, int port) {
        final StringBuilder id = new StringBuilder(String.format(Locale.US, "usb:%04x:%04x",
                device.getVendorId(), device.getProductId()));
        if (unit > 0) id.append('@').append(unit);
        if (port > 0) id.append('#').append(port);
        return id.toString();
    }

    private static int usbSuffix(String id, char marker) {
        final int at = id.indexOf(marker);
        if (at < 0) return 0;
        int end = at + 1;
        while (end < id.length() && Character.isDigit(id.charAt(end))) end++;
        try {
            return Integer.parseInt(id.substring(at + 1, end));
        } catch (NumberFormatException | StringIndexOutOfBoundsException e) {
            return 0;
        }
    }

    private static int usbPortIndex(String id) { return usbSuffix(id, '#'); }

    private static int usbUnitIndex(String id) { return usbSuffix(id, '@'); }

    /** `usb:VVVV:PPPP` with both suffixes stripped. */
    private static String usbBase(String id) {
        int end = id.length();
        final int at = id.indexOf('@');
        if (at >= 0) end = Math.min(end, at);
        final int hash = id.indexOf('#');
        if (hash >= 0) end = Math.min(end, hash);
        return id.substring(0, end);
    }

    private static UsbDevice findUsbDevice(String id) {
        final UsbManager manager = usbManager();
        if (manager == null || id == null || !id.startsWith("usb:")) return null;
        final String base = usbBase(id);
        for (UsbDevice device : manager.getDeviceList().values()) {
            if (!usbId(device, 0, 0).equals(base)) continue;
            // **The unit index, not the first match.** Two devices can
            // share a VID:PID and only one of them may be the radio's
            // CI-V port.
            final List<UsbDevice> units =
                    usbUnits(manager, device.getVendorId(), device.getProductId());
            final int unit = usbUnitIndex(id);
            return unit < units.size() ? units.get(unit) : null;
        }
        return null;
    }

    private static UsbManager usbManager() {
        if (context == null) return null;
        return (UsbManager) context.getSystemService(Context.USB_SERVICE);
    }

    private static BluetoothAdapter bluetoothAdapter() {
        if (context == null) return null;
        final BluetoothManager manager =
                (BluetoothManager) context.getSystemService(Context.BLUETOOTH_SERVICE);
        return manager == null ? null : manager.getAdapter();
    }

    // --- links ------------------------------------------------------------

    /** What both transports have in common, which is all C++ ever sees. */
    private interface Link {
        /** Bytes read, 0 on timeout, -1 if the link is gone. */
        int read(byte[] dest, int length, int timeoutMs) throws IOException;

        void write(byte[] src, int length) throws IOException;

        void setDtr(boolean on) throws IOException;

        void setRts(boolean on) throws IOException;

        void close();
    }

    static int open(String id, int baud, int dataBits, int stopBits, int parity, int flow,
                    boolean dtr, boolean rts) throws IOException {
        if (context == null) throw new IOException("SerialBridge.init was never called");
        // RFCOMM has no modem control lines at all, so `dtr` and `rts`
        // are simply not offered to it -- rather than silently ignored
        // in a place that reads as if they applied.
        final Link link = id != null && id.startsWith("bt:")
                ? openBluetooth(id)
                : openUsb(id, baud, dataBits, stopBits, parity, flow, dtr, rts);
        final int token = nextToken.getAndIncrement();
        links.put(token, link);
        return token;
    }

    /** The chip's own account of itself, or "" for a link that has none. */
    static String describeStatus(int token) {
        final Link link = links.get(token);
        if (!(link instanceof UsbLink)) return "";
        return ((UsbLink) link).status();
    }

    static void close(int token) {
        final Link link = links.remove(token);
        if (link != null) link.close();
    }

    static int read(int token, byte[] dest, int length, int timeoutMs) throws IOException {
        final Link link = links.get(token);
        if (link == null) return -1;
        return link.read(dest, length, timeoutMs);
    }

    static void write(int token, byte[] src, int length) throws IOException {
        final Link link = links.get(token);
        if (link == null) throw new IOException("the serial link is closed");
        link.write(src, length);
    }

    static void setDtr(int token, boolean on) throws IOException {
        final Link link = links.get(token);
        if (link == null) throw new IOException("the serial link is closed");
        link.setDtr(on);
    }

    static void setRts(int token, boolean on) throws IOException {
        final Link link = links.get(token);
        if (link == null) throw new IOException("the serial link is closed");
        link.setRts(on);
    }

    // --- USB --------------------------------------------------------------

    private static Link openUsb(String id, int baud, int dataBits, int stopBits, int parity,
                                int flow, boolean dtr, boolean rts) throws IOException {
        final UsbManager manager = usbManager();
        if (manager == null) throw new IOException("no USB service");
        final UsbDevice device = findUsbDevice(id);
        if (device == null) throw new IOException("no such USB device: " + id);
        if (!manager.hasPermission(device)) {
            // Deliberately not a permission request: this runs on the rig
            // worker thread, and a system dialog needs the UI thread and
            // an answer from a human. The settings screen asks; this only
            // reports.
            throw new IOException("no permission for " + id + " — grant it in Settings");
        }
        final UsbSerialDriver driver = UsbSerialProber.getDefaultProber().probeDevice(device);
        if (driver == null) throw new IOException("no serial driver for " + id);
        final int index = usbPortIndex(id);
        if (index >= driver.getPorts().size()) {
            throw new IOException("no port " + index + " on " + id);
        }
        final UsbSerialPort port = driver.getPorts().get(index);
        final UsbDeviceConnection connection = manager.openDevice(device);
        if (connection == null) {
            // The common cause on a composite audio+serial interface: the
            // platform's own driver has the device. Say so, because the
            // operator's next move is a different cable, not a different
            // setting.
            throw new IOException("could not open " + id
                    + " — another driver may hold it");
        }
        port.open(connection);
        try {
            port.setParameters(baud, dataBits, stopBits, toParity(parity));
            // **Before the flow control, deliberately.** The CP210x
            // SET_FLOW structure encodes whether each line is held
            // active, held inactive or driven by handshaking, and this
            // library builds it from the driver's current `dtr`/`rts`
            // fields — so setting the lines afterwards would leave the
            // chip's flow configuration describing the old state.
            //
            // And they are asserted by default because that is what a
            // serial port looks like everywhere else: the OS raises
            // both on open and Hamlib relies on it (`src/rig.c`, "Needed
            // on Linux because the serial port driver sets RTS/DTR on
            // open"). A bridged transport reaches none of that, and an
            // IC-9700 handed two low lines never answered a single CAT
            // command that the same cable answered instantly from a
            // desktop.
            port.setDTR(dtr);
            port.setRTS(rts);
            applyFlowControl(port, flow);
        } catch (IOException | UnsupportedOperationException e) {
            port.close();
            throw new IOException("could not configure " + id + ": " + e.getMessage());
        }
        return new UsbLink(port, connection);
    }

    private static int toParity(int parity) {
        switch (parity) {
            case 1: return UsbSerialPort.PARITY_ODD;
            case 2: return UsbSerialPort.PARITY_EVEN;
            default: return UsbSerialPort.PARITY_NONE;
        }
    }

    private static void applyFlowControl(UsbSerialPort port, int flow) {
        final UsbSerialPort.FlowControl want;
        switch (flow) {
            case 1: want = UsbSerialPort.FlowControl.RTS_CTS; break;
            case 2: want = UsbSerialPort.FlowControl.XON_XOFF; break;
            default: want = UsbSerialPort.FlowControl.NONE; break;
        }
        // **"None" means write nothing, not write zeros.** Setting a
        // CP210x's flow control to NONE means sending it a 16-byte
        // structure of zeroes, and an IC-9700's CP2102N stopped
        // answering CI-V entirely when it got one — every control
        // transfer succeeded, every bulk write returned its length, and
        // not one byte ever came back. FT8TW drives the same radio on
        // the same phone with a copy of this driver that has no
        // SET_FLOW code in it at all. A chip that was opened fresh is
        // already in its configured default, which is no flow control,
        // so there is nothing here to undo. See the matching guard in
        // the vendored `Cp21xxSerialDriver.openInt`, which does this
        // write before we are ever asked.
        if (want == UsbSerialPort.FlowControl.NONE) return;

        try {
            // **Not fatal if the chip cannot do it.** Only some drivers
            // implement hardware handshaking, and the alternative to
            // carrying on is refusing to talk to a radio that would have
            // worked — CAT is a few bytes at a time and almost never
            // needs flow control at all.
            if (port.getSupportedFlowControl().contains(want)) {
                port.setFlowControl(want);
            } else {
                Log.w(TAG, "flow control " + want + " unsupported; leaving it off");
            }
        } catch (IOException | UnsupportedOperationException e) {
            Log.w(TAG, "could not set flow control: " + e);
        }
    }

    private static final class UsbLink implements Link {
        private final UsbSerialPort port;
        private final UsbDeviceConnection connection;
        private volatile boolean closed;

        UsbLink(UsbSerialPort port, UsbDeviceConnection connection) {
            this.port = port;
            this.connection = connection;
        }

        /**
         * What the chip says about itself, for the rig trace.
         *
         * <p>A bulk write returning its length means the bytes reached
         * the chip over USB — not that the UART clocked them out.
         * {@code GET_COMM_STATUS} is the chip's own account: an error
         * mask, a bitmask of reasons transmission is being held, and the
         * two queue depths.
         *
         * <p><b>The queue depths turned out to be the weak half.</b>
         * The caller reads this a second after the previous frame, by
         * which time {@code outQueue} is zero whether the chip sent the
         * bytes or dropped them, and {@code inQueue} is drained
         * continuously by the read thread, so it is zero even when the
         * radio does answer. {@code errors} and {@code hold} are latched
         * by the chip rather than sampled, and those are what carry
         * information — along with CTS, since a handshake holding the
         * transmitter off would show up there.
         *
         * <p>Never throws: it runs on the bridge's read thread inside a
         * session that is otherwise working.
         */
        String status() {
            final StringBuilder out = new StringBuilder();
            try {
                out.append("lines");
                for (UsbSerialPort.ControlLine line : port.getSupportedControlLines()) {
                    out.append(' ').append(line).append('=');
                    switch (line) {
                        case RTS: out.append(port.getRTS() ? 1 : 0); break;
                        case CTS: out.append(port.getCTS() ? 1 : 0); break;
                        case DTR: out.append(port.getDTR() ? 1 : 0); break;
                        case DSR: out.append(port.getDSR() ? 1 : 0); break;
                        case CD: out.append(port.getCD() ? 1 : 0); break;
                        case RI: out.append(port.getRI() ? 1 : 0); break;
                        default: out.append('?'); break;
                    }
                }
            } catch (IOException | RuntimeException e) {
                out.append(" (unavailable: ").append(e).append(')');
            }

            // CP210x GET_COMM_STATUS. Silicon Labs AN571: 19 bytes, four
            // little-endian u32 followed by three bytes.
            if (!(port.getDriver() instanceof Cp21xxSerialDriver)) return out.toString();
            try {
                final byte[] buf = new byte[19];
                final int n = connection.controlTransfer(0xc1, 0x10, 0, port.getPortNumber(),
                        buf, buf.length, 500);
                if (n != buf.length) {
                    out.append("; comm status unavailable (").append(n).append(')');
                    return out.toString();
                }
                out.append("; errors=0x").append(Integer.toHexString(le32(buf, 0)))
                   .append(" hold=0x").append(Integer.toHexString(le32(buf, 4)))
                   .append(" inQueue=").append(le32(buf, 8))
                   .append(" outQueue=").append(le32(buf, 12));
            } catch (RuntimeException e) {
                out.append("; comm status threw ").append(e);
            }
            return out.toString();
        }

        private static int le32(byte[] b, int at) {
            return (b[at] & 0xff) | ((b[at + 1] & 0xff) << 8)
                    | ((b[at + 2] & 0xff) << 16) | ((b[at + 3] & 0xff) << 24);
        }

        @Override
        public int read(byte[] dest, int length, int timeoutMs) throws IOException {
            if (closed) return -1;
            try {
                // The library honours the timeout itself (a bulk transfer
                // with a deadline), so no reader thread is needed here —
                // unlike Bluetooth below, whose streams have none.
                return port.read(dest, length, timeoutMs);
            } catch (IOException e) {
                if (closed) return -1;
                throw e;
            } catch (IllegalStateException e) {
                // What the library raises when the device is unplugged
                // mid-read. Not an IOException, and an uncaught one here
                // would come out of JNI as a crash rather than a rig
                // error.
                return -1;
            }
        }

        @Override
        public void write(byte[] src, int length) throws IOException {
            if (closed) throw new IOException("the serial link is closed");
            port.write(src, length, WRITE_TIMEOUT_MS);
        }

        @Override
        public void setDtr(boolean on) throws IOException {
            port.setDTR(on);
        }

        @Override
        public void setRts(boolean on) throws IOException {
            port.setRTS(on);
        }

        @Override
        public void close() {
            closed = true;
            try {
                port.close();
            } catch (IOException | RuntimeException e) {
                Log.w(TAG, "closing the USB port: " + e);
            }
        }
    }

    // --- Bluetooth --------------------------------------------------------

    private static Link openBluetooth(String id) throws IOException {
        if (!bluetoothPermitted()) {
            throw new IOException("no Bluetooth permission — grant it in Settings");
        }
        final BluetoothAdapter adapter = bluetoothAdapter();
        if (adapter == null) throw new IOException("no Bluetooth adapter");
        final String address = id.substring(3);
        final BluetoothDevice device;
        try {
            device = adapter.getRemoteDevice(address);
        } catch (IllegalArgumentException e) {
            throw new IOException("not a Bluetooth address: " + address);
        }
        try {
            // Discovery is expensive and slows a connection down badly if
            // it happens to be running; cancelling is what every RFCOMM
            // example does and it costs nothing when it is not.
            adapter.cancelDiscovery();
            final BluetoothSocket socket = device.createRfcommSocketToServiceRecord(SPP);
            socket.connect();
            return new BluetoothLink(socket);
        } catch (SecurityException e) {
            throw new IOException("no Bluetooth permission — grant it in Settings");
        }
    }

    /**
     * RFCOMM, with a reader thread because {@code InputStream} has no timeout.
     *
     * <p>{@code BluetoothSocket}'s input stream blocks indefinitely and offers
     * nothing to bound it, so the timeout has to come from somewhere else: a
     * thread doing the blocking read and a queue with a poll deadline. That is
     * the same shape as the audio layer's capture thread, and for the same
     * reason — the blocking read belongs on a thread that is allowed to block.
     */
    private static final class BluetoothLink implements Link {
        private final BluetoothSocket socket;
        private final OutputStream out;
        private final LinkedBlockingQueue<byte[]> queue = new LinkedBlockingQueue<>();
        private final Thread reader;
        private volatile boolean closed;
        private byte[] leftover;
        private int leftoverAt;

        BluetoothLink(BluetoothSocket socket) throws IOException {
            this.socket = socket;
            this.out = socket.getOutputStream();
            final InputStream in = socket.getInputStream();
            reader = new Thread(() -> {
                final byte[] buf = new byte[512];
                while (!closed) {
                    try {
                        final int n = in.read(buf);
                        if (n < 0) break;
                        if (n == 0) continue;
                        final byte[] chunk = new byte[n];
                        System.arraycopy(buf, 0, chunk, 0, n);
                        queue.add(chunk);
                    } catch (IOException e) {
                        break;
                    }
                }
                closed = true;
                // Wake a reader waiting on the queue rather than making it
                // sit out its timeout to discover the link is gone.
                queue.add(new byte[0]);
            }, "sstvae-bt-rig");
            reader.setDaemon(true);
            reader.start();
        }

        @Override
        public int read(byte[] dest, int length, int timeoutMs) throws IOException {
            if (leftover == null) {
                if (closed && queue.isEmpty()) return -1;
                final byte[] chunk;
                try {
                    chunk = queue.poll(timeoutMs, TimeUnit.MILLISECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return 0;
                }
                if (chunk == null) return 0;            // an idle radio
                if (chunk.length == 0) return -1;       // the reader gave up
                leftover = chunk;
                leftoverAt = 0;
            }
            final int n = Math.min(length, leftover.length - leftoverAt);
            System.arraycopy(leftover, leftoverAt, dest, 0, n);
            leftoverAt += n;
            if (leftoverAt >= leftover.length) leftover = null;
            return n;
        }

        @Override
        public void write(byte[] src, int length) throws IOException {
            if (closed) throw new IOException("the Bluetooth link is closed");
            out.write(src, 0, length);
            out.flush();
        }

        @Override
        public void setDtr(boolean on) throws IOException {
            throw new IOException("Bluetooth has no DTR line — use CAT or VOX for PTT");
        }

        @Override
        public void setRts(boolean on) throws IOException {
            throw new IOException("Bluetooth has no RTS line — use CAT or VOX for PTT");
        }

        @Override
        public void close() {
            closed = true;
            try {
                // Closing is what unblocks the reader thread's read(): it
                // has no timeout of its own, so nothing else would.
                socket.close();
            } catch (IOException | RuntimeException e) {
                Log.w(TAG, "closing the Bluetooth socket: " + e);
            }
        }
    }

    private static native void nativePermissionResult(String id, boolean granted);
}
