package org.cleverdomain.sstvae;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Exports a received picture to the shared gallery.
 *
 * <p>This is what makes receptions appear under <i>On this device</i> in
 * Google Photos, and it is the only mechanism that does. Photos builds
 * those collections entirely from {@code MediaStore.Images}, grouped by
 * {@code BUCKET_DISPLAY_NAME} — the parent folder — and only for folders
 * on the shared volume. App-private storage, where receptions actually
 * live, is never indexed, so no amount of writing files there produces a
 * collection. The folder name below <em>is</em> the collection title.
 *
 * <p><b>The private copy stays canonical and this is a mirror.</b> The
 * sidecar JSON is what answers "who sent this and how well did it come
 * through" (see {@code Session::save_reception}), and none of it
 * survives the trip: MediaStore has no column Photos will display. So
 * the gallery copy is deliberately provenance-free, the app's own
 * Pictures screen remains the place that knows, and a failure here costs
 * an export rather than a reception.
 *
 * <p><b>No permission is involved.</b> Since API 29 an app may always
 * insert its own media, and it is only ever <em>its own</em> — which is
 * also why {@code minSdk} is 29 rather than 28: the legacy path is
 * {@code WRITE_EXTERNAL_STORAGE} plus a {@code MediaScannerConnection}
 * scan, a second implementation and a runtime prompt, for one API level.
 */
public final class Gallery {
    /** The bucket name, and therefore the collection title in Photos. */
    private static final String FOLDER = "SSTVAE";

    private Gallery() {}

    /**
     * Copies {@code path} into {@code Pictures/SSTVAE}.
     *
     * <p>Synchronous and does real I/O — roughly a megabyte per
     * reception — so callers must not run it on the main thread.
     *
     * @return null on success, or a short message suitable for showing
     *     to the operator.
     */
    public static String save(Context context, String path) {
        if (context == null || path == null) return "no context";
        final File file = new File(path);
        if (!file.exists()) return "the picture is gone";

        final ContentResolver resolver = context.getContentResolver();
        final ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, file.getName());
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.RELATIVE_PATH,
                Environment.DIRECTORY_PICTURES + File.separator + FOLDER);
        // **No DATE_TAKEN here, and it is not an oversight.** Setting it
        // on the insert looks right and is measured not to survive: once
        // IS_PENDING clears, the scanner re-derives the metadata columns
        // from the file itself, and a PNG carries no EXIF date, so the
        // column comes back NULL whatever was written. What is left is
        // `date_added`/`date_modified`, which is the moment the reception
        // finished — the right answer anyway. If a timeline position ever
        // has to be forced, it has to go *into the file* as metadata, not
        // into the row.
        //
        // **IS_PENDING is not optional.** Without it the media scanner
        // can index the row while the bytes are still being written, and
        // Photos then
        // shows a truncated picture that never repairs itself — the
        // entry is complete as far as it is concerned.
        values.put(MediaStore.Images.Media.IS_PENDING, 1);

        Uri uri = null;
        try {
            uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
            if (uri == null) return "the gallery refused the picture";

            try (InputStream in = new FileInputStream(file);
                 OutputStream out = resolver.openOutputStream(uri)) {
                if (out == null) throw new java.io.IOException("no output stream");
                final byte[] buf = new byte[64 * 1024];
                int n;
                while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
            }

            values.clear();
            values.put(MediaStore.Images.Media.IS_PENDING, 0);
            resolver.update(uri, values, null, null);
            return null;
        } catch (Exception e) {
            // Drop the pending row rather than leaving it. A pending
            // entry is invisible to other apps but it is not nothing: it
            // holds the display name, so the next export of the same
            // reception would be uniquified around a row that is only
            // wreckage.
            if (uri != null) {
                try {
                    resolver.delete(uri, null, null);
                } catch (Exception ignored) {
                    // Nothing useful left to do, and the caller is
                    // already being told the export failed.
                }
            }
            final String msg = e.getMessage();
            return (msg == null || msg.isEmpty()) ? e.getClass().getSimpleName() : msg;
        }
    }
}
