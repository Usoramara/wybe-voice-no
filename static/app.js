// Wybe Voice NO — WebSocket client with mic capture and audio playback

const MsgType = {
    HANDSHAKE: 0x01,
    AUDIO_IN:  0x02,
    AUDIO_OUT: 0x03,
    TEXT_ASR:  0x04,
    TEXT_LLM:  0x05,
    VAD_EVENT: 0x06,
    ERROR:     0x07,
    STATUS:    0x08,
};

let ws = null;
let mediaRecorder = null;
let audioStream = null;
let isRecording = false;
let audioCtx = null;
let currentAssistantEl = null;
let currentAssistantText = "";

const statusBar = document.getElementById("status-bar");
const transcript = document.getElementById("transcript");
const micBtn = document.getElementById("mic-btn");
const micLabel = document.getElementById("mic-label");

// --- WebSocket ---

function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/conversation`;
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        setStatus("ready");
        micBtn.disabled = false;
        // Send handshake
        const handshake = JSON.stringify({ version: 1 });
        const buf = new Uint8Array(1 + handshake.length);
        buf[0] = MsgType.HANDSHAKE;
        new TextEncoder().encodeInto(handshake, buf.subarray(1));
        ws.send(buf);
    };

    ws.onmessage = (evt) => {
        const data = new Uint8Array(evt.data);
        const type = data[0];
        const payload = data.slice(1);

        switch (type) {
            case MsgType.AUDIO_OUT:
                playAudio(payload);
                break;
            case MsgType.TEXT_ASR:
                handleASR(JSON.parse(new TextDecoder().decode(payload)));
                break;
            case MsgType.TEXT_LLM:
                handleLLM(JSON.parse(new TextDecoder().decode(payload)));
                break;
            case MsgType.VAD_EVENT:
                handleVAD(JSON.parse(new TextDecoder().decode(payload)));
                break;
            case MsgType.STATUS:
                handleStatus(JSON.parse(new TextDecoder().decode(payload)));
                break;
            case MsgType.ERROR:
                handleError(JSON.parse(new TextDecoder().decode(payload)));
                break;
        }
    };

    ws.onclose = () => {
        setStatus("Frakoblet", "error");
        micBtn.disabled = true;
        setTimeout(connect, 3000);
    };

    ws.onerror = () => {
        setStatus("Tilkoblingsfeil", "error");
    };
}

// --- Audio Recording ---

async function startRecording() {
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });
    } catch (e) {
        setStatus("Mikrofon avvist", "error");
        return;
    }

    mediaRecorder = new MediaRecorder(audioStream, {
        mimeType: "audio/webm;codecs=opus",
    });

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0 && ws && ws.readyState === WebSocket.OPEN) {
            e.data.arrayBuffer().then((buf) => {
                const payload = new Uint8Array(buf);
                const msg = new Uint8Array(1 + payload.length);
                msg[0] = MsgType.AUDIO_IN;
                msg.set(payload, 1);
                ws.send(msg);
            });
        }
    };

    // Send chunks every 200ms for responsive VAD
    mediaRecorder.start(200);
    isRecording = true;
    micBtn.classList.add("active");
    micLabel.textContent = "Lytter...";
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    if (audioStream) {
        audioStream.getTracks().forEach((t) => t.stop());
        audioStream = null;
    }
    isRecording = false;
    micBtn.classList.remove("active");
    micLabel.textContent = "Trykk for a snakke";
}

// --- Audio Playback ---

function playAudio(pcmData) {
    if (!audioCtx) {
        audioCtx = new AudioContext({ sampleRate: 24000 });
    }

    // PCM s16le → float32
    const int16 = new Int16Array(pcmData.buffer, pcmData.byteOffset, pcmData.byteLength / 2);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768;
    }

    const buffer = audioCtx.createBuffer(1, float32.length, 24000);
    buffer.getChannelData(0).set(float32);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);
    source.start();
}

// --- Message Handlers ---

function handleASR(data) {
    addMessage("user", data.text);
    // Prepare for assistant response
    currentAssistantText = "";
    currentAssistantEl = null;
}

function handleLLM(data) {
    if (data.done) return;

    currentAssistantText += data.text;

    if (!currentAssistantEl) {
        currentAssistantEl = addMessage("assistant", currentAssistantText);
    } else {
        currentAssistantEl.querySelector(".content").textContent = currentAssistantText;
    }
    transcript.scrollTop = transcript.scrollHeight;
}

function handleVAD(data) {
    if (data.event === "speech_start") {
        setStatus("Lytter...", "listening");
    }
}

function handleStatus(data) {
    const labels = {
        ready: "Klar",
        listening: "Lytter...",
        thinking: "Tenker...",
        speaking: "Snakker...",
    };
    setStatus(labels[data.status] || data.status, data.status);
}

function handleError(data) {
    setStatus("Feil: " + data.error, "error");
}

// --- UI Helpers ---

function setStatus(text, cls) {
    statusBar.textContent = text;
    statusBar.className = cls || "";
}

function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = `message ${role}`;

    const label = document.createElement("div");
    label.className = "label";
    label.textContent = role === "user" ? "Du" : "Wybe";

    const content = document.createElement("div");
    content.className = "content";
    content.textContent = text;

    div.appendChild(label);
    div.appendChild(content);
    transcript.appendChild(div);
    transcript.scrollTop = transcript.scrollHeight;
    return div;
}

// --- Init ---

micBtn.addEventListener("click", () => {
    // Resume AudioContext on user gesture (browser requirement)
    if (audioCtx && audioCtx.state === "suspended") {
        audioCtx.resume();
    }
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

connect();
