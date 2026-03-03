import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { api } from "../../../scripts/api.js";


const TBG_GGUF_INSTALL_POPUP_SEEN = new Set();

function tbgNormalizeExecutionError(event) {
  const detail = event?.detail || {};
  const message = String(detail.exception_message || "");
  const trace = Array.isArray(detail.traceback) ? detail.traceback.join("\n") : String(detail.traceback || "");
  const text = `${message}\n${trace}`.toLowerCase();
  return {
    key: `${detail.prompt_id || ""}|${detail.node_id || ""}|${message}` ,
    text,
  };
}

function tbgLooksLikeMissingLlamaCpp(text) {
  return text.includes("llama-cpp-python") && (text.includes("missing dependency") || text.includes("modulenotfounderror") || text.includes("no module named"));
}

async function tbgInstallOptionalGgufRuntime() {
  const advisory = "Optional GGUF install also sets up local llama-cpp server runtime dependencies. If you already have a llama-cpp/OpenAI-compatible server running, it is recommended to use that existing server through TBG ETUR Labs (OpenAI-Compatible setup) instead of running a second local server.";
  const confirmed = window.confirm(
    "GGUF runtime is missing (llama-cpp-python).\n\nInstall optional GGUF runtime now?\n\n" +
      advisory +
      "\n\nPress OK to install, Cancel to skip."
  );
  if (!confirmed) return;

  try {
    const response = await fetch(api.apiURL("/TBG/install_gguf_runtime"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const data = await response.json();
    const manual = data?.manual_command ? `\nManual command:\n${data.manual_command}` : "";
    const restart = data?.requires_restart ? "\n\nRestart ComfyUI to use GGUF." : "";
    window.alert(`${data?.message || "GGUF install request finished."}${restart}${manual}`);
  } catch (err) {
    window.alert(`GGUF runtime install request failed: ${err}`);
  }
}

api.addEventListener("execution_error", async (event) => {
  const normalized = tbgNormalizeExecutionError(event);
  if (!tbgLooksLikeMissingLlamaCpp(normalized.text)) return;
  if (TBG_GGUF_INSTALL_POPUP_SEEN.has(normalized.key)) return;
  TBG_GGUF_INSTALL_POPUP_SEEN.add(normalized.key);
  await tbgInstallOptionalGgufRuntime();
});

app.registerExtension({
  name: "TBG ETUR Upscaler and Tile Generator",

  init() {
    const STRING = ComfyWidgets.STRING;
    ComfyWidgets.STRING = function (node, inputName, inputData) {
      const r = STRING.apply(this, arguments);
      if (r?.widget && inputData?.[1]?.dynamicPrompts !== undefined) {
        r.widget.dynamicPrompts = inputData[1].dynamicPrompts;
      }
      return r;
    };
  },

  // ComfyUI 0.4.0 frontend still calls this, extra args are fine and ignored
  beforeRegisterNodeDef(nodeType /*, nodeData, appInstance */) {
    const cls = nodeType.comfyClass;

    // ------------------------------------------------------------------
    // Upscaler: show ui.value[0] (userinfo) as text on the node
    // ------------------------------------------------------------------
    if (cls === "TBG ETUR Upscaler and Tile Generator PRO" ||
    cls === "TBG ETUR Upscaler and Tile Generator CE"
    ) {
      const onDrawForeground = nodeType.prototype.onDrawForeground;



    // 2. Add DOM widget at bottom of node web panel
const onNodeCreated = nodeType.prototype.onNodeCreated;
nodeType.prototype.onNodeCreated = function () {
    // Call any previous onNodeCreated logic
    onNodeCreated?.apply?.(this, arguments);
    // --- Force seed control_after_generate to "fixed" on node create,
    // --- without hiding the selector in the UI.
    if (this.widgets && Array.isArray(this.widgets)) {
        // Find the control_after_generate widget created by ComfyUI
        const ctrlWidget = this.widgets.find(
            (w) => w && w.name === "control_after_generate"
        );

        if (ctrlWidget) {
            ctrlWidget.value = "fixed"; // or "randomize", "increment", "decrement"
            // Redraw node so UI shows the updated value
            this.onResize?.(this.size);
        }
    }

    // Container for the web panel (inside the node)
    const container = document.createElement("div");
    // Important: relative, not fixed; height matches iframe + padding
    container.style.cssText = [
        "position: relative",
        "padding: 0 4px 4px",
        "width: 100%",
        "height: 180px",            // fixed height to match iframe
        "box-sizing: border-box",
        "overflow: hidden"          // keep iframe visually inside node
    ].join("; ");

    // Optional text label (top of iframe area)
    const label = document.createElement("div");
    label.style.cssText = [
        "font-size: 11px",
        "color: #ccc",
        "margin-bottom: 2px",
        "white-space: nowrap",
        "overflow: hidden",
        "text-overflow: ellipsis"
    ].join("; ");
    container.appendChild(label);



    // DYNAMIC IFRAME src from v.value[1] (member/non-member) or v.value[0]
   const iframe = document.createElement("iframe");
    iframe.style.cssText = [
        "width: 100%",
        "height: 160px",   // 160 container - ~20 label/padding
        "border: 0px solid #444",
        "border-radius: 0px",
        "background: rgb(53, 53, 53)",
        "display: block"
    ].join("; ");    iframe.tabIndex = -1;
    iframe.sandbox = "allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation";
    iframe.loading = "lazy"; // Perf

    if (cls == "TBG ETUR Upscaler and Tile Generator PRO") {
      iframe.src = 'https://news.tbgetur.com/TBG_ETUR_News.html?type=PRO';
    } else {
      iframe.src = 'https://news.tbgetur.com/TBG_ETUR_News.html?type=CE';
    }




    container.appendChild(iframe);

    // Show always + LIVE UPDATE FUNCTION: Refresh text/iframe on node output changes
    let timeoutId;

    // Show always + LIVE UPDATE FUNCTION: Refresh text/iframe on node output changes
const updatePanel = () => {
    try {
        const v = app.nodeOutputs?.[this.id + ""];
        if (!v?.value) return;

        // Text from v.value[0] (credits info)
        const text = String(v.value[0] ?? "");
        console.log(`[TBG] Node ${this.id} Panel update - text: ${text}`);

        // Sanitize: Use textContent for label to prevent XSS, truncate for UI Units Left
        label.textContent = `${text.substring(0, 50)}`;

        // Iframe src: v.value[1] URL or fallback v.value[0]
        const src = v.value[1] || v.value[0] || "about:blank";
        console.log(`[TBG] Node ${this.id} iframe.src: ${src}`);  // ← This will print now!

        if (iframe.src !== src) {
            iframe.src = src;
        }
    } catch (e) {
        console.error("[TBG] Panel update error:", e);
    }
};

// Initial update
updatePanel();

// Hook ComfyUI's native node output changes (works when Python returns ui values)
const origOnExecuted = this.onExecuted;
this.onExecuted = function (output) {
    updatePanel();  // ← Runs every time node executes + outputs ui
    if (origOnExecuted) origOnExecuted.call(this, output);
};

// Optional: Listen to graph changes too (for loaded workflows)
//app.registerExtension({
//    name: "TBG.NodeOutputWatcher",
//    nodeCreated: (node) => {
//        if (node.comfyClass?.includes("TBG")) {
//            node.addEventListener("nodeExecuted", updatePanel);
//        }
//    }
//});

    // Initial update
    updatePanel();

    // Listen for node output changes (queue/prompt updates) with debounce
    const observer = new MutationObserver(() => {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(updatePanel, 500);
    });

    // Use a real DOM node as observer target
    const observeTarget =
        app?.canvas ||
        document.querySelector("#graph-canvas") ||
        document.body;

    if (observeTarget instanceof Node) {
        observer.observe(observeTarget, { childList: true, subtree: true });
    } else {
        console.warn("[TBG] No valid MutationObserver target; panel will only update on initial run.");
    }

    // Assemble & attach (iframe already in linkRow)
    //this.addDOMWidget("TBG Web Panel", "div", container, { serialize: false });

// === At the end of onNodeCreated, BEFORE any onRemoved override ===

// Attach the DOM widget; we do NOT try to touch internal widget.options
this.addDOMWidget("TBG Web Panel", "div", container, {
    serialize: false
});

// Option 1: grow node height explicitly so the dom-widget area is tall enough
const PANEL_EXTRA = 195; // extra vertical space you want to reserve for the iframe
const currentWidth = this.size?.[0] ?? 250;
const currentHeight = this.size?.[1] ?? 80;

// Ensure the node is tall enough to visually contain the panel + existing controls
const targetHeight = Math.max(currentHeight, PANEL_EXTRA);
this.setSize([currentWidth, targetHeight]);

// Option 2 (optional): add 3–4 empty widgets to push height via standard rows
// This is your “4–5 empty widgets” idea, implemented safely.
for (let i = 0; i < 1; i++) {
    const w = this.addWidget("text", "", "", () => {}, {
        serialize: false,
    });
    const TALL_H = 130;
w.computeSize = function () {
    // width is ignored for layout, only height matters here
    return [this.parent?.size?.[0] ?? 200, TALL_H];
};
}
// Make w widget taller than default

// Request a canvas redraw so the new size takes effect
this.setDirtyCanvas(true, true);


    // Cleanup on node remove without clobbering other hooks
    const prevOnRemoved = this.onRemoved;
    this.onRemoved = function () {
        try {
            observer.disconnect();
        } catch (e) {
            console.error("[TBG] Observer disconnect error:", e);
        }
        if (typeof prevOnRemoved === "function") {
            return prevOnRemoved.apply(this, arguments);
        }
    };
};
}






    // ------------------------------------------------------------------
    // Magnific variant, unchanged from your code
    // ------------------------------------------------------------------
    if (cls === "TBG ETUR Magnific Magnifier") {
      const onDrawForeground = nodeType.prototype.onDrawForeground;

      nodeType.prototype.onDrawForeground = function (ctx) {
        const r = onDrawForeground?.apply?.(this, arguments);

        const v = app.nodeOutputs?.[this.id + ""];
        if (!this.flags.collapsed && v && v.value && v.value[0] != null) {
          const text = v.value[0] + "";
          ctx.save();
          // ctx.font = "6px";
          // ctx.fillStyle = "dodgerblue";
          const sz = ctx.measureText(text);
          // ctx.fillText(text, this.size[0] - sz.width - 5, -15);
          ctx.fillText(text, 50, 218);
          ctx.restore();
        }

        return r;
      };
    }

    // ------------------------------------------------------------------
    // Refiner: add "last seed used" + Copy button, update from ui.value[0]
    // ------------------------------------------------------------------
    if (
      cls === "TBG ETUR Refiner PRO"
    ) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;

      nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply?.(this, arguments);
    // --- Force seed control_after_generate to "fixed" on node create,
    // --- without hiding the selector in the UI.
    if (this.widgets && Array.isArray(this.widgets)) {
        // Find the control_after_generate widget created by ComfyUI
        const ctrlWidget = this.widgets.find(
            (w) => w && w.name === "control_after_generate"
        );

        if (ctrlWidget) {
            ctrlWidget.value = "fixed"; // or "randomize", "increment", "decrement"
            // Redraw node so UI shows the updated value
            this.onResize?.(this.size);
        }
    }

        // Container
        const container = document.createElement("div");
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.gap = "4px";
        container.style.padding = "4px";
        container.style.marginBottom = "8px";

        // Label
        const label = document.createElement("div");
        label.innerText = "last seed used";
        label.style.fontSize = "12px";
        label.style.color = "#ccc";
        label.style.marginBottom = "2px";

        // Row (input + button)
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "4px";

        // Seed field
        const seedField = document.createElement("input");
        seedField.type = "text";
        seedField.readOnly = true;
        seedField.style.flex = "1";
        seedField.style.background = "#444";
        seedField.style.color = "#eee";
        seedField.style.border = "1px solid #666";
        seedField.style.borderRadius = "999px";
        seedField.style.padding = "4px 8px";
        seedField.style.fontSize = "12px";

        // Copy button
        const copyBtn = document.createElement("button");
        copyBtn.innerText = "Copy";
        copyBtn.style.background = "#555";
        copyBtn.style.color = "#eee";
        copyBtn.style.border = "1px solid #666";
        copyBtn.style.borderRadius = "999px";
        copyBtn.style.padding = "4px 10px";
        copyBtn.style.cursor = "pointer";
        copyBtn.style.fontSize = "12px";
        copyBtn.onmouseenter = () => (copyBtn.style.background = "#666");
        copyBtn.onmouseleave = () => (copyBtn.style.background = "#555");
        copyBtn.onclick = () => {
          if (seedField.value) navigator.clipboard.writeText(seedField.value);
        };

        // Assemble
        row.appendChild(seedField);
        row.appendChild(copyBtn);
        container.appendChild(row);
        container.appendChild(label);

        const spacer = document.createElement("div");
        spacer.style.height = "8px";
        container.appendChild(spacer);

        this.addDOMWidget("Seed", "div", container, { serialize: false });
        this.seedField = seedField;

        // If graph was loaded from a run that already has ui.value, pre-fill:
        const v = app.nodeOutputs?.[this.id + ""];
        if (v && v.value && v.value[0] != null) {
          this.seedField.value = String(v.value[0]);
        }
      };

      const onExecuted = nodeType.prototype.onExecuted;

      nodeType.prototype.onExecuted = function (output) {
        const r = onExecuted?.apply?.(this, arguments);

        // On 0.4.0, ui from Python fn() is on app.nodeOutputs[id].value [file:59][web:41][web:50]
        const v = app.nodeOutputs?.[this.id + ""];
        if (v && v.value && v.value[0] != null && this.seedField) {
          this.seedField.value = String(v.value[0]);
        }

        return r;
      };
    }
  },
});
