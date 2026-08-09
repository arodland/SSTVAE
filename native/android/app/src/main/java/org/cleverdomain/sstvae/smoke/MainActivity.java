package org.cleverdomain.sstvae.smoke;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.BitmapFactory;
import android.media.AudioManager;
import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.json.JSONObject;

import java.io.File;
import java.util.List;

/**
 * The smoke test's whole UI: pick an input, start, watch it decode.
 *
 * <p>Explicitly <em>not</em> the Tier 0 interface (see docs/android.md). There
 * is no service, so this dies with the activity; there is no gallery, no
 * waterfall, and no notification. Those are the things Tier 0 is actually
 * about, and building them here would be building the wrong app to answer the
 * question this one exists for — can a phone select an audio device and decode
 * a picture.
 */
public final class MainActivity extends AppCompatActivity {

    private Spinner deviceSpinner;
    private Button startButton;
    private TextView statusText;
    private ImageView pictureView;

    private CaptureThread capture;
    private WavFeeder feeder;
    private List<AudioDevices.Entry> devices;
    private String shownPicture = "";

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final Runnable poll = new Runnable() {
        @Override
        public void run() {
            refresh();
            ui.postDelayed(this, 500);
        }
    };

    private ActivityResultLauncher<String> micPermission;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // targetSdk 35+ enforces edge-to-edge, so the content window is the
        // whole screen and anything at the top draws *under* the status bar
        // and the action bar. Without this the device spinner is hidden
        // behind the title -- which looked like the spinner failing to
        // populate rather than a layout problem, and is worth the six lines
        // even in a smoke test.
        View root = findViewById(R.id.root);
        ViewCompat.setOnApplyWindowInsetsListener(root, (v, windowInsets) -> {
            Insets bars = windowInsets.getInsets(
                    WindowInsetsCompat.Type.systemBars()
                            | WindowInsetsCompat.Type.displayCutout());
            int pad = (int) (16 * getResources().getDisplayMetrics().density);
            v.setPadding(bars.left + pad, bars.top + pad, bars.right + pad,
                    bars.bottom + pad);
            return WindowInsetsCompat.CONSUMED;
        });

        deviceSpinner = findViewById(R.id.devices);
        startButton = findViewById(R.id.start);
        statusText = findViewById(R.id.status);
        pictureView = findViewById(R.id.picture);

        micPermission = registerForActivityResult(
                new ActivityResultContracts.RequestPermission(), granted -> {
                    if (granted) {
                        startCapture();
                    } else {
                        Toast.makeText(this, "Microphone permission is required",
                                Toast.LENGTH_LONG).show();
                    }
                });

        startButton.setOnClickListener(v -> {
            if (capture != null) {
                stopCapture();
            } else if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                    == PackageManager.PERMISSION_GRANTED) {
                startCapture();
            } else {
                micPermission.launch(Manifest.permission.RECORD_AUDIO);
            }
        });

        // Long-press START to feed a WAV instead of opening the microphone.
        // Not a UI worth defending; it is how the decode path gets exercised
        // where the emulator's virtual mic cannot carry a signal.
        startButton.setOnLongClickListener(v -> {
            if (capture != null || feeder != null) return true;
            if (modelMissing()) return true;
            File wav = new File("/data/local/tmp/tx48.wav");
            if (!wav.exists()) {
                Toast.makeText(this, "no " + wav, Toast.LENGTH_LONG).show();
                return true;
            }
            feeder = new WavFeeder(wav, msg -> ui.post(() -> {
                Toast.makeText(MainActivity.this, msg, Toast.LENGTH_LONG).show();
                feeder = null;
            }));
            feeder.start();
            startButton.setText("Feeding");
            return true;
        });

        statusText.setOnLongClickListener(v -> {
            String path = new File(App.filesDirectory(), "ring.wav").getAbsolutePath();
            String err = Native.dumpAudio(path);
            Toast.makeText(this, err.isEmpty() ? ("dumped " + path) : err,
                    Toast.LENGTH_LONG).show();
            return true;
        });

        refreshDevices();
        ui.post(poll);
    }

    private void refreshDevices() {
        devices = AudioDevices.inputs(this);
        ArrayAdapter<AudioDevices.Entry> adapter = new ArrayAdapter<>(
                this, android.R.layout.simple_spinner_item, devices);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        deviceSpinner.setAdapter(adapter);
    }

    private boolean modelMissing() {
        if (!Native.hasCodec() || App.foundModelDir() != null) return false;
        statusText.setText("No decoder model.\nPut " + App.DECODER + " in\n"
                + App.preferredModelDir()
                + "\n\nadb push <file> " + App.preferredModelDir() + "/");
        Toast.makeText(this, "No decoder model - see the status text",
                Toast.LENGTH_LONG).show();
        return true;
    }

    private void startCapture() {
        if (modelMissing()) return;
        int position = deviceSpinner.getSelectedItemPosition();
        int id = position >= 0 && position < devices.size() ? devices.get(position).id : 0;
        AudioManager am = (AudioManager) getSystemService(Context.AUDIO_SERVICE);

        capture = new CaptureThread(am, id, new CaptureThread.Listener() {
            @Override
            public void onOpened(int rate, int channels, String format) {
                ui.post(() -> statusText.setText("Opened at " + rate + " Hz"));
            }

            @Override
            public void onError(String message) {
                ui.post(() -> Toast.makeText(MainActivity.this, message,
                        Toast.LENGTH_LONG).show());
            }
        });
        capture.start();
        startButton.setText("Stop");
        deviceSpinner.setEnabled(false);
    }

    private void stopCapture() {
        if (feeder != null) {
            feeder.shutdown();
            feeder = null;
        }
        if (capture != null) {
            capture.shutdown();
            capture = null;
        }
        Native.stop();
        startButton.setText("Start");
        deviceSpinner.setEnabled(true);
    }

    private void refresh() {
        // Nothing to overwrite the message with, and overwriting it is
        // exactly the bug the desktop's three error tiers exist to stop:
        // a sticky explanation destroyed by a routine status line.
        if (capture == null && feeder == null && Native.hasCodec()
                && App.foundModelDir() == null) {
            return;
        }
        String json = Native.status();
        try {
            JSONObject o = new JSONObject(json);
            StringBuilder sb = new StringBuilder();
            sb.append(o.optString("status", "Idle"));
            sb.append("   polls ").append(o.optInt("polls"));
            sb.append("   ring ").append(String.format("%.1f", o.optDouble("ring_s", 0)))
                    .append(" s");
            if (o.optBoolean("resampling")) sb.append("   [resampling]");
            sb.append('\n');

            int frames = o.optInt("frames");
            int expected = o.optInt("expected");
            if (frames > 0) {
                sb.append("frames ").append(frames);
                if (expected > 0) sb.append('/').append(expected);
                sb.append("   ");
            }
            String mode = o.optString("mode", "");
            if (!mode.isEmpty()) sb.append("mode ").append(mode).append("   ");
            String callsign = o.optString("callsign", "");
            if (!callsign.isEmpty()) sb.append(callsign).append("   ");
            if (!o.isNull("snr")) {
                sb.append(String.format("SNR %.1f dB", o.optDouble("snr")));
            }
            sb.append("\ncompleted ").append(o.optInt("completed"));

            // The audio side, shown while capturing. A level in dBFS plus the
            // share of near-silent samples, because those are different
            // questions: "is it too quiet" and "is anything arriving at all"
            // have the same mean and are not the same problem.
            CaptureThread c = capture;
            if (c != null && !c.openedAs.isEmpty()) {
                sb.append("\nin: ").append(c.openedAs)
                        .append("  ->  ").append(c.routedTo);
                int pk = c.levelPeak;
                String db = pk > 0
                        ? String.format("%.0f dBFS", 20 * Math.log10(pk / 32768.0))
                        : "silent";
                sb.append(String.format("%n    peak %s   %.1f%% near-zero",
                        db, c.levelNearZeroPct));
                if (!c.routingWarning.isEmpty()) {
                    sb.append("\n    ! ").append(c.routingWarning);
                }
            }
            if (!Native.hasCodec()) sb.append("   (no codec: frames only)");

            String error = o.optString("error", "");
            if (!error.isEmpty()) sb.append("\nERROR: ").append(error);
            statusText.setText(sb.toString());

            String saved = o.optString("saved", "");
            if (!saved.isEmpty() && !saved.equals(shownPicture)) {
                File f = new File(saved);
                if (f.exists()) {
                    pictureView.setImageBitmap(BitmapFactory.decodeFile(saved));
                    shownPicture = saved;
                }
            }
        } catch (Exception e) {
            statusText.setText(json);
        }
    }

    @Override
    protected void onDestroy() {
        ui.removeCallbacks(poll);
        stopCapture();
        super.onDestroy();
    }
}
