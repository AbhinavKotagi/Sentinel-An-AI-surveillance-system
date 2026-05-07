/**
 * SENTINEL — Real-Time Dashboard Script
 * 
 * Polls /status every 1 second and updates the DOM.
 * Manages dynamic alert feed with latest alerts prepended.
 * Handles backend offline gracefully.
 */

const API_BASE = "http://localhost:5000";
const POLL_INTERVAL = 1000; // 1 second

// ─── State tracking ──────────────────────────────────────────────────────────
let previousThreat = "SAFE";
let isBackendOnline = false;
let alertCount = 0;

// ─── Threat level config ─────────────────────────────────────────────────────
const THREAT_CONFIG = {
    SAFE: {
        emoji: "🟢",
        color: "text-green-400",
        borderColor: "border-green-500/30",
        bgColor: "bg-green-500/5",
        barColor: "bg-green-500",
        label: "LEVEL 0",
        description: "All clear — No threats detected",
        icon: "lucide:shield-check",
    },
    MEDIUM: {
        emoji: "🟡",
        color: "text-yellow-500",
        borderColor: "border-yellow-500/30",
        bgColor: "bg-yellow-500/5",
        barColor: "bg-yellow-500",
        label: "LEVEL 1",
        description: "Elevated activity in sector",
        icon: "lucide:alert-triangle",
    },
    HIGH: {
        emoji: "🟠",
        color: "text-orange-500",
        borderColor: "border-orange-500/30",
        bgColor: "bg-orange-500/5",
        barColor: "bg-orange-500",
        label: "LEVEL 2",
        description: "Active threat in sector",
        icon: "lucide:alert-triangle",
    },
    CRITICAL: {
        emoji: "🔴",
        color: "text-[#ef233c]",
        borderColor: "border-[#ef233c]/30",
        bgColor: "bg-[#ef233c]/5",
        barColor: "bg-[#ef233c]",
        label: "LEVEL 3",
        description: "Critical threat — Immediate action required",
        icon: "lucide:alert-triangle",
    },
};

// ─── DOM element cache ───────────────────────────────────────────────────────
const elements = {};

function cacheElements() {
    elements.threatLevel = document.getElementById("threatLevel");
    elements.threatLevelBadge = document.getElementById("threatLevelBadge");
    elements.threatLevelLabel = document.getElementById("threatLevelLabel");
    elements.threatDescription = document.getElementById("threatDescription");
    elements.threatIcon = document.getElementById("threatIcon");
    elements.threatDot = document.getElementById("threatDot");
    elements.personsCount = document.getElementById("personsCount");
    elements.weaponStatus = document.getElementById("weaponStatus");
    elements.weaponCard = document.getElementById("weaponCard");
    elements.motionScore = document.getElementById("motionScore");
    elements.movementScore = document.getElementById("movementScore");
    elements.alertFeed = document.getElementById("alertFeed");
    elements.cameraFeed = document.getElementById("cameraFeed");
    elements.systemClock = document.getElementById("systemClock");
    elements.cameraStatus = document.getElementById("cameraStatus");
    elements.aiConfidence = document.getElementById("aiConfidence");
    elements.objectsDetected = document.getElementById("objectsDetected");
    elements.motionScoreOverlay = document.getElementById("motionScoreOverlay");
    elements.fpsCounter = document.getElementById("fpsCounter");
    elements.footerThreat = document.getElementById("footerThreat");
}

// ─── Update system clock ─────────────────────────────────────────────────────
function updateClock() {
    if (elements.systemClock) {
        const now = new Date();
        const formatted = now.getFullYear() + "." +
            String(now.getMonth() + 1).padStart(2, "0") + "." +
            String(now.getDate()).padStart(2, "0") + " — " +
            String(now.getHours()).padStart(2, "0") + ":" +
            String(now.getMinutes()).padStart(2, "0") + ":" +
            String(now.getSeconds()).padStart(2, "0") + " GMT";
        elements.systemClock.textContent = formatted;
    }
}

// ─── Update threat status panel ──────────────────────────────────────────────
function updateThreatPanel(threat) {
    const config = THREAT_CONFIG[threat] || THREAT_CONFIG.SAFE;

    if (elements.threatLevel) {
        elements.threatLevel.textContent = threat;
        // Remove all threat color classes, add new one
        elements.threatLevel.className = elements.threatLevel.className
            .replace(/text-\[#ef233c\]|text-green-\d+|text-yellow-\d+|text-orange-\d+/g, "")
            .trim();
        // Set color via style for reliability
        const colorMap = { SAFE: "#22c55e", MEDIUM: "#eab308", HIGH: "#f97316", CRITICAL: "#ef233c" };
        elements.threatLevel.style.color = colorMap[threat] || "#ef233c";
    }

    if (elements.threatLevelLabel) {
        elements.threatLevelLabel.textContent = config.label;
    }

    if (elements.threatDescription) {
        elements.threatDescription.textContent = config.description;
    }

    if (elements.threatIcon) {
        elements.threatIcon.setAttribute("icon", config.icon);
        const iconColor = { SAFE: "#22c55e", MEDIUM: "#eab308", HIGH: "#f97316", CRITICAL: "#ef233c" };
        elements.threatIcon.style.color = iconColor[threat] || "#ef233c";
    }

    if (elements.threatDot) {
        const dotColor = { SAFE: "bg-green-500", MEDIUM: "bg-yellow-500", HIGH: "bg-orange-500", CRITICAL: "bg-[#ef233c]" };
        elements.threatDot.className = `w-2 h-2 rounded-full animate-pulse ${dotColor[threat] || "bg-[#ef233c]"}`;
    }

    // Update the threat card border glow
    const threatCard = document.getElementById("threatCard");
    if (threatCard) {
        if (threat === "SAFE") {
            threatCard.classList.remove("alert-active");
            threatCard.style.borderColor = "rgba(34, 197, 94, 0.3)";
            threatCard.style.boxShadow = "0 0 20px rgba(34, 197, 94, 0.15)";
        } else if (threat === "MEDIUM") {
            threatCard.classList.remove("alert-active");
            threatCard.style.borderColor = "rgba(234, 179, 8, 0.3)";
            threatCard.style.boxShadow = "0 0 20px rgba(234, 179, 8, 0.15)";
        } else if (threat === "HIGH") {
            threatCard.classList.remove("alert-active");
            threatCard.style.borderColor = "rgba(249, 115, 22, 0.3)";
            threatCard.style.boxShadow = "0 0 20px rgba(249, 115, 22, 0.15)";
        } else {
            threatCard.classList.add("alert-active");
            threatCard.style.borderColor = "";
            threatCard.style.boxShadow = "";
        }
    }
}

// ─── Update detection metrics ────────────────────────────────────────────────
function updateMetrics(data) {
    if (elements.personsCount) {
        elements.personsCount.textContent = data.persons;
    }

    if (elements.weaponStatus) {
        const weaponText = data.weapon ? "YES" : "NO";
        elements.weaponStatus.textContent = weaponText;
        if (data.weapon) {
            elements.weaponStatus.style.color = "#ef233c";
        } else {
            elements.weaponStatus.style.color = "#22c55e";
        }
    }

    // Weapon card border highlight
    if (elements.weaponCard) {
        if (data.weapon) {
            elements.weaponCard.classList.add("border", "border-[#ef233c]/30");
        } else {
            elements.weaponCard.classList.remove("border-[#ef233c]/30");
        }
    }

    if (elements.motionScore) {
        elements.motionScore.textContent = data.motion + "%";
    }

    if (elements.movementScore) {
        elements.movementScore.textContent = data.movement + "%";
    }

    // Camera overlay stats
    if (elements.aiConfidence) {
        elements.aiConfidence.textContent = data.ai_confidence + "%";
    }
    if (elements.objectsDetected) {
        elements.objectsDetected.textContent = data.objects_detected;
    }
    if (elements.motionScoreOverlay) {
        const motionLabel = data.motion > 65 ? "HIGH" : data.motion > 35 ? "MEDIUM" : "LOW";
        elements.motionScoreOverlay.textContent = motionLabel;
        const motionColors = { HIGH: "#facc15", MEDIUM: "#f97316", LOW: "#22c55e" };
        elements.motionScoreOverlay.style.color = motionColors[motionLabel];
    }
    if (elements.fpsCounter) {
        elements.fpsCounter.textContent = "FPS: " + data.fps;
    }

    // Footer threat level
    if (elements.footerThreat) {
        elements.footerThreat.textContent = "// THREAT_LEVEL: " + data.threat;
    }
}

// ─── Update camera status overlay ────────────────────────────────────────────
function updateCameraStatus(threat) {
    if (elements.cameraStatus) {
        if (threat === "SAFE") {
            elements.cameraStatus.textContent = "Live • Monitoring";
            elements.cameraStatus.style.color = "#22c55e";
        } else {
            elements.cameraStatus.textContent = "Live • Threat Detected";
            elements.cameraStatus.style.color = "#ef233c";
        }
    }

    // Camera feed border
    const cameraCard = document.getElementById("cameraCard");
    if (cameraCard) {
        if (threat === "SAFE") {
            cameraCard.style.borderColor = "rgba(255,255,255,0.05)";
            cameraCard.style.boxShadow = "none";
        } else if (threat === "CRITICAL") {
            cameraCard.style.borderColor = "#ef233c";
            cameraCard.style.boxShadow = "0 0 20px rgba(239,35,60,0.3)";
        } else if (threat === "HIGH") {
            cameraCard.style.borderColor = "rgba(249,115,22,0.5)";
            cameraCard.style.boxShadow = "0 0 15px rgba(249,115,22,0.2)";
        } else {
            cameraCard.style.borderColor = "rgba(234,179,8,0.3)";
            cameraCard.style.boxShadow = "0 0 10px rgba(234,179,8,0.15)";
        }
    }
}

// ─── Dynamic alert feed ──────────────────────────────────────────────────────
function addAlertToFeed(level, message, time) {
    if (!elements.alertFeed) return;

    const config = THREAT_CONFIG[level] || THREAT_CONFIG.SAFE;

    // Build alert HTML that matches the existing design exactly
    let alertHTML = "";

    if (level === "CRITICAL") {
        alertHTML = `
            <div class="p-3 bg-[#ef233c]/5 border border-[#ef233c]/20 rounded-lg relative overflow-hidden">
                <div class="absolute left-0 top-0 bottom-0 w-1 bg-[#ef233c]"></div>
                <div class="flex items-start justify-between mb-1">
                    <div class="flex items-center gap-2">
                        <span class="text-sm">${config.emoji}</span>
                        <span class="text-[10px] font-bold text-[#ef233c] uppercase">${level}</span>
                    </div>
                    <span class="text-[9px] font-mono text-zinc-500">${time}</span>
                </div>
                <p class="text-[11px] text-white leading-relaxed">${message}</p>
            </div>`;
    } else if (level === "HIGH") {
        alertHTML = `
            <div class="p-3 bg-orange-500/5 border border-orange-500/20 rounded-lg relative overflow-hidden">
                <div class="absolute left-0 top-0 bottom-0 w-1 bg-orange-500"></div>
                <div class="flex items-start justify-between mb-1">
                    <div class="flex items-center gap-2">
                        <span class="text-sm">${config.emoji}</span>
                        <span class="text-[10px] font-bold text-orange-500 uppercase">${level}</span>
                    </div>
                    <span class="text-[9px] font-mono text-zinc-500">${time}</span>
                </div>
                <p class="text-[11px] text-zinc-300 leading-relaxed">${message}</p>
            </div>`;
    } else if (level === "MEDIUM") {
        alertHTML = `
            <div class="p-3 hover:bg-white/5 rounded-lg transition-all">
                <div class="flex items-start justify-between mb-1">
                    <div class="flex items-center gap-2">
                        <span class="text-sm">${config.emoji}</span>
                        <span class="text-[10px] font-bold text-yellow-500/80 uppercase">${level}</span>
                    </div>
                    <span class="text-[9px] font-mono text-zinc-500">${time}</span>
                </div>
                <p class="text-[11px] text-zinc-400 leading-relaxed">${message}</p>
            </div>`;
    } else {
        alertHTML = `
            <div class="p-3 hover:bg-white/5 rounded-lg transition-all">
                <div class="flex items-start justify-between mb-1">
                    <div class="flex items-center gap-2">
                        <span class="text-sm">${config.emoji}</span>
                        <span class="text-[10px] font-bold text-green-500/80 uppercase">${level}</span>
                    </div>
                    <span class="text-[9px] font-mono text-zinc-500">${time}</span>
                </div>
                <p class="text-[11px] text-zinc-500 leading-relaxed">${message}</p>
            </div>`;
    }

    // Prepend (latest on top)
    elements.alertFeed.insertAdjacentHTML("afterbegin", alertHTML);

    // Keep max 20 alerts in the DOM
    while (elements.alertFeed.children.length > 20) {
        elements.alertFeed.removeChild(elements.alertFeed.lastChild);
    }
}

// ─── Fetch alerts from backend ───────────────────────────────────────────────
async function fetchAlerts() {
    try {
        const res = await fetch(API_BASE + "/alerts");
        if (!res.ok) return;
        const alerts = await res.json();

        // Only add new alerts (compare count)
        if (alerts.length > alertCount) {
            const newAlerts = alerts.slice(0, alerts.length - alertCount);
            // Add in reverse order so latest ends up on top
            for (let i = newAlerts.length - 1; i >= 0; i--) {
                addAlertToFeed(newAlerts[i].level, newAlerts[i].message, newAlerts[i].time);
            }
            alertCount = alerts.length;
        }
    } catch (err) {
        // Silently ignore alert fetch errors
    }
}

// ─── Main fetch loop ─────────────────────────────────────────────────────────
async function fetchStatus() {
    try {
        const res = await fetch(API_BASE + "/status");
        if (!res.ok) throw new Error("HTTP " + res.status);
        const data = await res.json();

        if (!isBackendOnline) {
            isBackendOnline = true;
            console.log("[SENTINEL] Backend connected.");
        }

        // Update all UI sections
        updateThreatPanel(data.threat);
        updateMetrics(data);
        updateCameraStatus(data.threat);

        previousThreat = data.threat;
    } catch (err) {
        if (isBackendOnline) {
            isBackendOnline = false;
            console.warn("[SENTINEL] Backend offline:", err.message);
        }
    }
}

// ─── Initialise ──────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    cacheElements();
    updateClock();

    // Start polling
    setInterval(fetchStatus, POLL_INTERVAL);
    setInterval(fetchAlerts, 2000);  // Check alerts every 2s
    setInterval(updateClock, 1000);

    // Initial fetch
    fetchStatus();
    fetchAlerts();

    console.log("[SENTINEL] Dashboard initialised. Polling backend at", API_BASE);
});
