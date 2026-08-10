package org.cleverdomain.sstvae;

import android.app.Activity;
import android.content.ContentResolver;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import androidx.core.content.FileProvider;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Choosing a picture to transmit, from the gallery or the camera.
 *
 * <p><b>Its own transparent activity, rather than Qt's.</b> An activity
 * result has to come back to the activity that launched it, and reaching
 * {@code QtActivity}'s means either subclassing it or Qt's private
 * {@code QtAndroidPrivate::startActivity}. Subclassing puts this app in
 * the business of maintaining a Qt bindings subclass across upgrades, and
 * the private API is private. A three-screen activity that launches an
 * intent and finishes is smaller than either and depends on nothing that
 * can move underneath it.
 *
 * <p><b>The result is copied into app-private storage before anyone sees
 * a path.</b> What the picker returns is a {@code content://} URI with a
 * grant that lasts as long as this activity does — so handing the URI to
 * C++ would produce a path that reads fine during composition and fails
 * at the moment of transmission. Copying is a few megabytes once, against
 * a failure mode that only shows up when it costs an over.
 */
public class ImagePicker extends Activity {
    private static final String TAG = "SSTVAE";
    private static final String EXTRA_MODE = "mode";
    private static final String MODE_PICK = "pick";
    private static final String MODE_CAPTURE = "capture";
    private static final int REQUEST = 7001;

    /** Where a camera app is asked to write. Held across the round trip
     *  because the result intent for a capture carries no data at all —
     *  the agreement is that the photo went where we said. */
    private File captureTarget;

    private static native void nativePicked(String path, String error);

    public static void pick(Context context) {
        launch(context, MODE_PICK);
    }

    public static void capture(Context context) {
        launch(context, MODE_CAPTURE);
    }

    private static void launch(Context context, String mode) {
        if (context == null) return;
        Intent i = new Intent(context, ImagePicker.class);
        i.setAction(Intent.ACTION_MAIN);
        i.putExtra(EXTRA_MODE, mode);
        i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        context.startActivity(i);
    }

    @Override
    protected void onCreate(Bundle saved) {
        super.onCreate(saved);
        // A relaunch after the process was killed mid-pick has no result
        // to deliver and no target to deliver it to. Finishing is the
        // honest outcome; retrying would reopen a chooser the operator
        // did not ask for a second time.
        if (saved != null) {
            finish();
            return;
        }

        final String mode = getIntent() != null ? getIntent().getStringExtra(EXTRA_MODE) : null;
        try {
            startActivityForResult(MODE_CAPTURE.equals(mode) ? captureIntent() : pickIntent(),
                    REQUEST);
        } catch (Exception e) {
            Log.e(TAG, "could not start the picker", e);
            nativePicked(null, "No app on this phone can supply a picture that way.");
            finish();
        }
    }

    /**
     * {@code ACTION_OPEN_DOCUMENT} rather than {@code ACTION_GET_CONTENT}
     * or the photo picker: it is the one that works on every API this app
     * supports (28 up) and returns something readable without a storage
     * permission.
     */
    private Intent pickIntent() {
        Intent i = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        i.addCategory(Intent.CATEGORY_OPENABLE);
        i.setType("image/*");
        return i;
    }

    private Intent captureIntent() throws Exception {
        // The *external* cache, because Qt's FileProvider covers it and
        // the internal one is not in its path list. Declaring a second
        // provider would collide with Qt's, which is the same trap
        // Sharing.java records.
        File dir = getExternalCacheDir();
        if (dir == null) throw new IllegalStateException("no external cache directory");
        captureTarget = new File(dir, "capture.jpg");
        Uri out = FileProvider.getUriForFile(this, getPackageName() + ".qtprovider",
                captureTarget);
        Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        i.putExtra(MediaStore.EXTRA_OUTPUT, out);
        i.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        return i;
    }

    @Override
    protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != REQUEST) {
            finish();
            return;
        }
        if (result != RESULT_OK) {
            // Cancelling is not an error and must not raise one: backing
            // out of the chooser is the most ordinary thing an operator
            // does there. Null on both arguments says "nothing changed".
            nativePicked(null, null);
            finish();
            return;
        }

        try {
            if (captureTarget != null && (data == null || data.getData() == null)) {
                nativePicked(captureTarget.getAbsolutePath(), null);
            } else {
                nativePicked(copyIn(data.getData()).getAbsolutePath(), null);
            }
        } catch (Exception e) {
            Log.e(TAG, "could not read the chosen picture", e);
            nativePicked(null, "Could not read that picture.");
        }
        finish();
    }

    /** Copy a content:// URI into app-private storage and return the file. */
    private File copyIn(Uri uri) throws Exception {
        if (uri == null) throw new IllegalArgumentException("no picture was returned");
        File out = new File(getCacheDir(), "compose-source");
        ContentResolver cr = getContentResolver();
        try (InputStream in = cr.openInputStream(uri);
             OutputStream os = new FileOutputStream(out)) {
            if (in == null) throw new IllegalStateException("could not open that picture");
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) os.write(buf, 0, n);
        }
        return out;
    }
}
