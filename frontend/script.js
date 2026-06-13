/* Yaseen AI — frontend logic */
document.addEventListener("DOMContentLoaded", () => {
    const chat = document.getElementById("chat");
    const chatScroll = document.getElementById("chat-scroll");
    const intro = document.getElementById("intro");
    const chips = document.getElementById("chips");
    const input = document.getElementById("input");
    const send = document.getElementById("send");
    const composer = document.getElementById("composer");
    const typing = document.getElementById("typing");
    const statusDot = document.getElementById("status-dot");

    const API_CHAT = "/chat";
    const API_HEALTH = "/health";
    const MAX_HISTORY = 16; // messages kept client-side (server trims again)

    let history = [];
    let busy = false;

    /* ---- server status ---------------------------------------------- */
    async function checkHealth() {
        try {
            const res = await fetch(API_HEALTH);
            const data = await res.json();
            statusDot.classList.toggle("ok", data.status === "ok");
            statusDot.classList.toggle("error", data.status !== "ok");
        } catch {
            statusDot.classList.add("error");
        }
    }
    checkHealth();

    /* ---- rendering ---------------------------------------------------- */
    function addMessage(role, text) {
        const div = document.createElement("div");
        div.className = `msg ${role}`;
        div.textContent = text; // textContent prevents XSS
        chat.appendChild(div);
        chatScroll.scrollTop = chatScroll.scrollHeight;
        return div;
    }

    function setBusy(state) {
        busy = state;
        send.disabled = state;
        typing.hidden = !state;
    }

    /* ---- chat flow ----------------------------------------------------- */
    async function sendMessage(text) {
        const message = (text ?? input.value).trim();
        if (!message || busy) return;

        if (intro) intro.remove();

        addMessage("user", message);
        input.value = "";
        autosize();
        setBusy(true);

        try {
            const res = await fetch(API_CHAT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message, history }),
            });

            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(data.error || `Request failed (HTTP ${res.status}).`);
            }

            addMessage("bot", data.response);
            history.push({ role: "user", content: message });
            history.push({ role: "assistant", content: data.response });
            if (history.length > MAX_HISTORY) {
                history = history.slice(-MAX_HISTORY);
            }
        } catch (err) {
            addMessage("error", `Something went wrong: ${err.message}`);
        } finally {
            setBusy(false);
            input.focus();
        }
    }

    /* ---- input handling ------------------------------------------------ */
    function autosize() {
        input.style.height = "auto";
        input.style.height = Math.min(input.scrollHeight, 140) + "px";
    }

    composer.addEventListener("submit", (e) => {
        e.preventDefault();
        sendMessage();
    });

    input.addEventListener("input", autosize);

    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    if (chips) {
        chips.addEventListener("click", (e) => {
            const chip = e.target.closest(".chip");
            if (chip) sendMessage(chip.textContent);
        });
    }

    input.focus();
});
