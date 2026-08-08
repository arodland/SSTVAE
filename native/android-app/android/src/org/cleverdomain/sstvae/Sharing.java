package org.cleverdomain.sstvae;

import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import androidx.core.content.FileProvider;

import java.io.File;

/**
 * Hands a received picture to the system share sheet.
 *
 * <p>Receptions live in app-private storage, so the file path itself is
 * meaningless to any other app — a {@code content://} URI with a
 * one-shot read grant is the only way out. Qt already declares the
 * provider this needs ({@code ${applicationId}.qtprovider}) and its
 * {@code files-path} covers {@code getFilesDir()}, which is where the
 * gallery writes, so **no second provider is required**. Declaring one
 * would have been the obvious move and would have collided with Qt's.
 *
 * <p>The share sheet rather than a MediaStore insert, deliberately: it
 * reaches everything — gallery, mail, chat, a file manager — for one
 * intent and no storage permission, where writing to the shared gallery
 * is a narrower destination for more code. Both can exist; this is the
 * one that makes a picture useful immediately.
 */
public final class Sharing {
    private Sharing() {}

    public static void share(Context context, String path, String caption) {
        if (context == null || path == null) return;
        final File file = new File(path);
        if (!file.exists()) return;

        final Uri uri = FileProvider.getUriForFile(
                context, context.getPackageName() + ".qtprovider", file);

        final Intent send = new Intent(Intent.ACTION_SEND);
        send.setType("image/png");
        send.putExtra(Intent.EXTRA_STREAM, uri);
        if (caption != null && !caption.isEmpty()) {
            // The metadata travels with the picture. On the receiving
            // side "who sent this and how well did it come through" is
            // the same question the sidecar exists to answer.
            send.putExtra(Intent.EXTRA_TEXT, caption);
        }
        send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        // **The grant has to travel on the ClipData, not only on
        // EXTRA_STREAM.** A flag alone covers the intent's own data
        // URI, and this one is in an extra, so the system has nothing
        // to attach the grant to -- the target app gets it when it is
        // launched, but the chooser's own thumbnail loader does not,
        // and fails with `Permission Denial: opening provider ...
        // FileProvider`. Visible as a share sheet with a broken preview
        // for a share that then works, which is a confusing half
        // failure to be handed.
        send.setClipData(ClipData.newUri(context.getContentResolver(), "reception", uri));

        final Intent chooser = Intent.createChooser(send, "Share reception");
        // `createChooser` *wraps* the intent, so the flags above are on
        // the inner one and the resolver process is not covered by
        // them. It needs its own copy.
        chooser.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        // The chooser is started from whatever context we were handed,
        // which may not be an Activity on every path.
        chooser.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        context.startActivity(chooser);
    }
}
