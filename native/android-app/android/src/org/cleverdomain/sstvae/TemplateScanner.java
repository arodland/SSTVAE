package org.cleverdomain.sstvae;

import android.content.Context;
import android.util.Log;

import com.google.mlkit.common.MlKitException;
import com.google.mlkit.vision.barcode.common.Barcode;
import com.google.mlkit.vision.codescanner.GmsBarcodeScanner;
import com.google.mlkit.vision.codescanner.GmsBarcodeScannerOptions;
import com.google.mlkit.vision.codescanner.GmsBarcodeScanning;

import java.nio.charset.StandardCharsets;

/**
 * Reading a template off another screen: the desktop's Share window
 * shows one as a QR code, and this hands what the camera read to
 * {@code Transmitter::importTemplate}, the same slot a paste goes to.
 *
 * <p><b>ML Kit's code scanner, unbundled.</b> Google Play services owns
 * the camera, the preview and the decoder; this app declares no camera
 * permission, ships no model, and gains no APK size. The trade is that
 * the scanner module lives in Play services and arrives from the network
 * the first time any app on the phone asks for it -- so it can be absent
 * on a phone that has never scanned anything, briefly, and the paste path
 * is what covers that. The manifest asks for it at install time
 * ({@code com.google.mlkit.vision.DEPENDENCIES}), which makes that case
 * rare rather than merely recoverable.
 *
 * <p><b>The raw bytes, decoded as UTF-8, not {@code getRawValue()}.</b>
 * The desktop encodes the payload's UTF-8 in byte mode; the scanner's
 * string form guesses the charset from the bytes, and a guess is not what
 * a callsign with an accent in a Comment deserves. The bytes are exactly
 * what was encoded.
 */
public final class TemplateScanner {
    private static final String TAG = "SSTVAE";

    /** Both null: the operator backed out. An error alone: what went wrong. */
    private static native void nativeScanned(String payload, String error);

    private TemplateScanner() {}

    public static void scan(Context context) {
        if (context == null) {
            nativeScanned(null, "No screen to scan from.");
            return;
        }
        GmsBarcodeScannerOptions options = new GmsBarcodeScannerOptions.Builder()
                // Only QR: the scanner would otherwise happily read a
                // product barcode on the desk as a "template".
                .setBarcodeFormats(Barcode.FORMAT_QR_CODE)
                // A 73-module code on a monitor is dense from across a
                // desk; auto-zoom is what makes it read at arm's length.
                .enableAutoZoom()
                .build();
        GmsBarcodeScanner scanner = GmsBarcodeScanning.getClient(context, options);
        scanner.startScan()
                .addOnSuccessListener(barcode -> nativeScanned(payloadOf(barcode), null))
                .addOnCanceledListener(() -> nativeScanned(null, null))
                .addOnFailureListener(e -> {
                    Log.w(TAG, "template scan failed", e);
                    nativeScanned(null, describe(e));
                });
    }

    private static String payloadOf(Barcode barcode) {
        byte[] raw = barcode.getRawBytes();
        if (raw != null) return new String(raw, StandardCharsets.UTF_8);
        return barcode.getRawValue();
    }

    /** A message for the operator, or null for a cancellation that arrived
     *  as an exception rather than through the cancel listener. */
    static String describe(Exception e) {
        if (e instanceof MlKitException) {
            switch (((MlKitException) e).getErrorCode()) {
                case MlKitException.CODE_SCANNER_CANCELLED:
                    return null;
                case MlKitException.UNAVAILABLE:
                case MlKitException.CODE_SCANNER_UNAVAILABLE:
                    return "Google Play services is still installing its scanner. "
                            + "Try again in a moment, or paste the text instead.";
                case MlKitException.CODE_SCANNER_GOOGLE_PLAY_SERVICES_VERSION_TOO_OLD:
                    return "Google Play services is too old to scan. "
                            + "Update it, or paste the text instead.";
                case MlKitException.CODE_SCANNER_CAMERA_PERMISSION_NOT_GRANTED:
                    return "The scanner was not allowed to use the camera.";
                default:
                    break;
            }
        }
        String why = e.getMessage();
        return "Could not scan" + (why == null || why.isEmpty() ? "." : ": " + why);
    }
}
