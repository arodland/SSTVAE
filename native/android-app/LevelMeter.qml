import QtQuick

// The input level, as a bar rather than a number.
//
// **Scaled in dBFS, not linearly.** A linear bar spends most of its
// width on the top 20 dB and squashes everything an operator actually
// tunes by into the leftmost pixels; a receiver is set up somewhere
// around -20 dBFS, which on a linear scale is a tenth of the way along
// and indistinguishable from silence.
//
// The zones encode the only three judgements that matter here. **Too
// quiet** is the common real fault (squelch closed, wrong input,
// nothing plugged in). **Good** is a deliberately wide band, because
// this modem does not need a carefully set level — it is scale
// invariant, and implying otherwise would send people hunting for a
// precision that buys nothing. **Clipping** is the one that costs
// picture quality, so it is the only zone drawn in red.
Item {
    id: root

    property real peak: 0.0        // 0..1 linear
    property bool dropping: false
    // Numbers when the technical switch is on, a judgement when it is
    // off. The bar and its colour are unchanged either way -- what is
    // hidden is the reading, not the meter, because "is my level all
    // right" is a question every operator has and "-23 dBFS" answers
    // it only for someone who already knows the answer.
    property bool technical: false

    implicitHeight: 26

    // **Qt hex colours are `#AARRGGBB`, not `#RRGGBBAA`.** Written the
    // CSS way, every alpha here landed in the red channel and the alpha
    // came out 0x00: the background, the border and the text were fully
    // transparent, and the good band -- meant to be a faint green --
    // painted as pale red. The result still *looked* like a widget, an
    // offset pink rectangle with no text, which is why it survived a
    // commit. It is only visible while listening, so no idle screenshot
    // shows it.

    readonly property real floorDb: -60
    readonly property real db: peak > 0 ? 20 * Math.log(peak) / Math.LN10 : floorDb
    readonly property real frac: Math.max(0, Math.min(1, (db - floorDb) / (0 - floorDb)))

    function xFor(d) { return width * Math.max(0, (d - floorDb) / (0 - floorDb)) }

    Rectangle {
        anchors.fill: parent
        radius: 3
        color: "#14000000"
        border.color: "#22000000"
    }

    // The good band, drawn as a backdrop so the bar reads against it.
    Rectangle {
        x: root.xFor(-30)
        width: root.xFor(-6) - x
        height: parent.height
        color: "#182e7d32"
    }

    Rectangle {
        id: bar
        x: 1; y: 1
        height: parent.height - 2
        width: Math.max(0, root.frac * (parent.width - 2))
        radius: 2
        color: root.dropping ? "#c62828"
             : root.db > -3   ? "#c62828"
             : root.db > -30  ? "#2e7d32"
             : "#ef6c00"
        Behavior on width { NumberAnimation { duration: 80 } }
    }

    // Clip marker: latched by the colour above only while it lasts, so
    // a brief overload during a whole transmission is easy to miss.
    // This tick is where 0 dBFS is, for reference rather than alarm.
    Rectangle {
        x: parent.width - 2
        width: 2
        height: parent.height
        color: "#33000000"
    }

    Text {
        anchors.centerIn: parent
        font.pixelSize: 11
        font.family: "monospace"
        color: "#aa000000"
        // The plain wording tracks exactly the same thresholds the bar
        // is coloured by, so the word and the colour can never
        // disagree. "Audio problem" for dropped samples rather than
        // anything more specific: the operator cannot act on ppm, and
        // the number is one switch away for whoever can.
        text: root.dropping ? (root.technical ? "DROPPING AUDIO" : "Audio problem")
            : root.peak <= 0 ? (root.technical ? "no signal" : "No audio")
            : root.technical ? root.db.toFixed(0) + " dBFS"
            : root.db > -3  ? "Too loud"
            : root.db > -30 ? "Level good"
            : "Quiet"
    }
}
