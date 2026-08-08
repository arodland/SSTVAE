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

    implicitHeight: 26

    readonly property real floorDb: -60
    readonly property real db: peak > 0 ? 20 * Math.log(peak) / Math.LN10 : floorDb
    readonly property real frac: Math.max(0, Math.min(1, (db - floorDb) / (0 - floorDb)))

    function xFor(d) { return width * Math.max(0, (d - floorDb) / (0 - floorDb)) }

    Rectangle {
        anchors.fill: parent
        radius: 3
        color: "#00000014"
        border.color: "#00000022"
    }

    // The good band, drawn as a backdrop so the bar reads against it.
    Rectangle {
        x: root.xFor(-30)
        width: root.xFor(-6) - x
        height: parent.height
        color: "#2e7d3218"
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
        color: "#00000033"
    }

    Text {
        anchors.centerIn: parent
        font.pixelSize: 11
        font.family: "monospace"
        color: "#000000aa"
        text: root.dropping ? "DROPPING AUDIO"
            : root.peak <= 0 ? "no signal"
            : root.db.toFixed(0) + " dBFS"
    }
}
