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
 * operator unplugs the radio. {@code usb:10c4:ea60} survives, with {@code #N}
 * for the second and later ports of a multi-port adapter. Two identical
 * adapters are indistinguishable and the first wins — the same accepted
 * ambiguity the audio layer takes on device names.
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
                nativePermissionResult(device == null ? "" : usbId(device, 0), granted);
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
        for (UsbSerialDriver driver : UsbSerialProber.getDefaultProber()
                .findAllDrivers(manager)) {
            final UsbDevice device = driver.getDevice();
            final boolean permitted = manager.hasPermission(device);
            final int ports = driver.getPorts().size();
            for (int i = 0; i < ports; i++) {
                final StringBuilder label = new StringBuilder(describe(device));
                if (ports > 1) label.append(" port ").append(i + 1);
                out.add(usbId(device, i) + "\t" + clean(label.toString()) + "\t"
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
            for (BluetoothDevice device : bonded) {
                final String name = device.getName();
                final String label = (name == null || name.isEmpty())
                        ? device.getAddress()
                        : name + " (" + device.getAddress() + ")";
                out.add("bt:" + device.getAddress() + "\t" + clean(label) + "\t1");
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

    private static String usbId(UsbDevice device, int port) {
        final String base = String.format(Locale.US, "usb:%04x:%04x",
                device.getVendorId(), device.getProductId());
        return port == 0 ? base : base + "#" + port;
    }

    private static int usbPortIndex(String id) {
        final int hash = id.indexOf('#');
        if (hash < 0) return 0;
        try {
            return Integer.parseInt(id.substring(hash + 1));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private static UsbDevice findUsbDevice(String id) {
        final UsbManager manager = usbManager();
        if (manager == null || id == null || !id.startsWith("usb:")) return null;
        final int hash = id.indexOf('#');
        final String base = hash < 0 ? id : id.substring(0, hash);
        for (UsbDevice device : manager.getDeviceList().values()) {
            if (usbId(device, 0).equals(base)) return device;
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

    static int open(String id, int baud, int dataBits, int stopBits, int parity, int flow)
            throws IOException {
        if (context == null) throw new IOException("SerialBridge.init was never called");
        final Link link = id != null && id.startsWith("bt:")
                ? openBluetooth(id)
                : openUsb(id, baud, dataBits, stopBits, parity, flow);
        final int token = nextToken.getAndIncrement();
        links.put(token, link);
        return token;
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
                                int flow) throws IOException {
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
            applyFlowControl(port, flow);
        } catch (IOException | UnsupportedOperationException e) {
            port.close();
            throw new IOException("could not configure " + id + ": " + e.getMessage());
        }
        return new UsbLink(port);
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
        try {
            // **Not fatal if the chip cannot do it.** Only some drivers
            // implement hardware handshaking, and the alternative to
            // carrying on is refusing to talk to a radio that would have
            // worked — CAT is a few bytes at a time and almost never
            // needs flow control at all.
            if (want == UsbSerialPort.FlowControl.NONE
                    || port.getSupportedFlowControl().contains(want)) {
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
        private volatile boolean closed;

        UsbLink(UsbSerialPort port) {
            this.port = port;
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
