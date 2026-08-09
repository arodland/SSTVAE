package org.cleverdomain.sstvae.smoke;

import android.content.Context;
import android.media.AudioDeviceInfo;
import android.media.AudioManager;

import java.util.ArrayList;
import java.util.List;

/**
 * Input device enumeration.
 *
 * <p>This is the reason the audio layer is Java rather than AAudio/Oboe.
 * {@link AudioManager#getDevices} is the only way to see what is attached, and
 * {@code AudioRecord.setPreferredDevice} is the only way to say which one to
 * use — AAudio's {@code setDeviceId} is honoured only when the underlying API
 * is AAudio and is silently ignored on the OpenSL ES fallback, and Qt's Android
 * backend is reported not to honour a device selection at all. A setting that
 * quietly does nothing is the failure mode this project has spent the most time
 * on, so it is worth avoiding structurally.
 *
 * <p>The name is what the config file would store, so it has to be stable and
 * human-readable — the same reasoning as {@code audio::match_device} on the
 * desktop, which matches descriptions rather than opaque backend ids. Note it
 * is not unique when two identical interfaces are attached; the desktop has the
 * same ambiguity and lives with it.
 */
final class AudioDevices {

    static final class Entry {
        final int id;
        final String name;

        Entry(int id, String name) {
            this.id = id;
            this.name = name;
        }

        @Override
        public String toString() {
            return name;
        }
    }

    private AudioDevices() {}

    static List<Entry> inputs(Context context) {
        AudioManager am = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
        List<Entry> out = new ArrayList<>();
        // id 0 is AudioRecord's "no preference", i.e. whatever the system
        // would route to anyway. Offered explicitly so the operator can tell
        // "the default" apart from a device that merely happens to be first.
        out.add(new Entry(0, "System default"));
        if (am == null) return out;
        for (AudioDeviceInfo d : am.getDevices(AudioManager.GET_DEVICES_INPUTS)) {
            out.add(new Entry(d.getId(), describe(d)));
        }
        return out;
    }

    /**
     * The type is part of the name on purpose: several phones report every
     * input with the same product name, and "USB" versus "Builtin Mic" is
     * precisely the distinction the operator is trying to make.
     */
    private static String describe(AudioDeviceInfo d) {
        String product = String.valueOf(d.getProductName()).trim();
        String type = typeName(d.getType());
        if (product.isEmpty()) return type;
        return product + " (" + type + ")";
    }

    static String describeType(int type) {
        return typeName(type);
    }

    private static String typeName(int type) {
        switch (type) {
            case AudioDeviceInfo.TYPE_BUILTIN_MIC:
                return "Builtin Mic";
            case AudioDeviceInfo.TYPE_USB_DEVICE:
                return "USB Device";
            case AudioDeviceInfo.TYPE_USB_HEADSET:
                return "USB Headset";
            case AudioDeviceInfo.TYPE_USB_ACCESSORY:
                return "USB Accessory";
            case AudioDeviceInfo.TYPE_WIRED_HEADSET:
                return "Wired Headset";
            case AudioDeviceInfo.TYPE_BLUETOOTH_SCO:
                return "Bluetooth SCO";
            case AudioDeviceInfo.TYPE_TELEPHONY:
                return "Telephony";
            case AudioDeviceInfo.TYPE_FM_TUNER:
                return "FM Tuner";
            default:
                return "Type " + type;
        }
    }
}
