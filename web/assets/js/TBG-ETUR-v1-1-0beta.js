import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
import { api } from "../../../scripts/api.js";
import {
  getOutputValue,
  isTBGNode,
  requestNodeRedraw,
  resolveTBGNodeClass,
  safeApply,
  setNodeMinHeight,
} from "./TBG-ETUR-compat.js";


const TBG_GGUF_INSTALL_POPUP_SEEN = new Set();

const TBG_UPSCALER_NODE_NAMES = [
  "TBG ETUR Upscaler and Tile Generator PRO",
  "TBG ETUR Upscaler and Tile Generator CE",
];

function tbgLooksLikeUrl(value) {
  const text = String(value ?? "").trim();
  return /^(https?:)?\/\//i.test(text) || text.startsWith("/") || text.startsWith("about:") || text.startsWith("blob:");
}

function tbgIsLegacyNodeMode() {
  try {
    return app?.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled") !== true;
  } catch (_) {
    return true;
  }
}

function tbgPanelWidgetType() {
  return tbgIsLegacyNodeMode() ? "customtext" : "div";
}

function tbgGetCanvasElement() {
  return app?.canvas?.canvas || document.querySelector("canvas");
}

function tbgCloneCanvasEvent(event) {
  const common = {
    bubbles: true,
    cancelable: true,
    composed: true,
    view: window,
    detail: event.detail || 0,
    screenX: event.screenX || 0,
    screenY: event.screenY || 0,
    clientX: event.clientX || 0,
    clientY: event.clientY || 0,
    ctrlKey: !!event.ctrlKey,
    shiftKey: !!event.shiftKey,
    altKey: !!event.altKey,
    metaKey: !!event.metaKey,
    button: event.button || 0,
    buttons: event.buttons || 0,
  };

  if (event instanceof WheelEvent) {
    return new WheelEvent(event.type, {
      ...common,
      deltaX: event.deltaX,
      deltaY: event.deltaY,
      deltaZ: event.deltaZ,
      deltaMode: event.deltaMode,
    });
  }

  if (window.PointerEvent && event instanceof PointerEvent) {
    return new PointerEvent(event.type, {
      ...common,
      pointerId: event.pointerId,
      width: event.width,
      height: event.height,
      pressure: event.pressure,
      tangentialPressure: event.tangentialPressure,
      tiltX: event.tiltX,
      tiltY: event.tiltY,
      twist: event.twist,
      pointerType: event.pointerType,
      isPrimary: event.isPrimary,
    });
  }

  return new MouseEvent(event.type, common);
}

function tbgForwardEventToCanvas(event) {
  const canvas = tbgGetCanvasElement();
  if (!canvas) return false;
  event.preventDefault();
  event.stopPropagation();
  canvas.dispatchEvent(tbgCloneCanvasEvent(event));
  return true;
}

function tbgAttachCanvasEventForwarding(rootEl) {
  const controller = new AbortController();
  const signal = controller.signal;

  rootEl.style.touchAction = "none";

  rootEl.addEventListener("wheel", (event) => {
    tbgForwardEventToCanvas(event);
  }, { passive: false, signal });

  rootEl.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
  }, { signal });

  rootEl.addEventListener("pointerdown", (event) => {
    if (event.button === 1 || event.button === 2) {
      tbgForwardEventToCanvas(event);
    }
  }, { capture: true, signal });

  rootEl.addEventListener("pointermove", (event) => {
    if ((event.buttons & 4) === 4 || (event.buttons & 2) === 2) {
      tbgForwardEventToCanvas(event);
    }
  }, { signal });

  rootEl.addEventListener("pointerup", (event) => {
    if (event.button === 1 || event.button === 2) {
      tbgForwardEventToCanvas(event);
    }
  }, { capture: true, signal });

  rootEl.addEventListener("auxclick", (event) => {
    if (event.button === 1) {
      event.preventDefault();
      event.stopPropagation();
    }
  }, { signal });

  return controller;
}

function tbgFitLegacyDomWidgetNode(node, minHeight) {
  if (!tbgIsLegacyNodeMode()) return;
  const computed = typeof node?.computeSize === "function" ? node.computeSize() : null;
  const computedHeight = Array.isArray(computed) ? computed[1] : 0;
  setNodeMinHeight(node, Math.max(minHeight, computedHeight || 0));
}

function tbgEnsureFixedControlAfterGenerate(node) {
  if (!Array.isArray(node?.widgets)) return;
  const ctrlWidget = node.widgets.find((w) => w && w.name === "control_after_generate");
  if (!ctrlWidget) return;
  ctrlWidget.value = "fixed";
  requestNodeRedraw(node);
}

function tbgUpdateInfoPanel(node, output) {
  const panel = node?.__tbgPanel;
  if (!panel?.label || !panel?.iframe) return;

  const value = getOutputValue(node, output);
  if (!value || value.length === 0) return;

  const text = String(value[0] ?? "");
  panel.label.textContent = text.substring(0, 50);

  const preferredSrc = String(value[1] ?? "").trim();
  const fallbackSrc = String(value[0] ?? "").trim();
  const nextSrc = preferredSrc || (tbgLooksLikeUrl(fallbackSrc) ? fallbackSrc : "");
  if (nextSrc && panel.iframe.src !== nextSrc) {
    panel.iframe.src = nextSrc;
  }
}

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
  beforeRegisterNodeDef(nodeType, nodeData /*, appInstance */) {
    const cls = resolveTBGNodeClass(nodeType, nodeData);

    // ------------------------------------------------------------------
    // Upscaler: show ui.value[0] (userinfo) as text on the node
    // ------------------------------------------------------------------
    if (isTBGNode(nodeType, nodeData, TBG_UPSCALER_NODE_NAMES)) {
      const onNodeCreated = nodeType.prototype.onNodeCreated;
      const onExecuted = nodeType.prototype.onExecuted;
      const onConfigure = nodeType.prototype.onConfigure;

      nodeType.prototype.onNodeCreated = function () {
        safeApply(onNodeCreated, this, arguments);
        tbgEnsureFixedControlAfterGenerate(this);

        const panelWidgetHeight = 190;
        const panelNodeMinHeight = 230;

        if (this.__tbgPanelInitialized) {
          tbgUpdateInfoPanel(this);
          tbgFitLegacyDomWidgetNode(this, panelNodeMinHeight);
          return;
        }

        const container = document.createElement("div");
        container.style.cssText = [
          "position: relative",
          "padding: 0 4px 4px",
          "width: 100%",
          "min-width: 0",
          "height: 190px",
          "min-height: 180px",
          "box-sizing: border-box",
          "overflow: hidden",
        ].join("; ");

        const label = document.createElement("div");
        label.style.cssText = [
          "font-size: 11px",
          "color: #ccc",
          "margin-bottom: 2px",
          "white-space: nowrap",
          "overflow: hidden",
          "text-overflow: ellipsis",
        ].join("; ");
        container.appendChild(label);

        const iframe = document.createElement("iframe");
        iframe.style.cssText = [
          "width: 100%",
          "height: 160px",
          "border: 0px solid #444",
          "border-radius: 0px",
          "background: rgb(53, 53, 53)",
          "display: block",
          "pointer-events: auto",
        ].join("; ");
        iframe.tabIndex = -1;
        iframe.sandbox = "allow-scripts allow-same-origin allow-popups allow-top-navigation-by-user-activation";
        iframe.loading = "lazy";
        iframe.scrolling = "auto";
        iframe.src = cls === "TBG ETUR Upscaler and Tile Generator PRO"
          ? "https://news.tbgetur.com/TBG_ETUR_News.html?type=PRO"
          : "https://news.tbgetur.com/TBG_ETUR_News.html?type=CE";
        container.appendChild(iframe);

        const eventController = tbgAttachCanvasEventForwarding(container);
        const widget = this.addDOMWidget("TBG Web Panel", tbgPanelWidgetType(), container, {
          serialize: false,
          getHeight() { return panelWidgetHeight; },
          getMinHeight() { return panelWidgetHeight; },
          getValue() { return ""; },
          setValue() {},
        });
        const prevWidgetOnRemove = widget?.onRemove;
        if (widget) {
          widget.onRemove = function () {
            try { eventController.abort(); } catch (_) {}
            return prevWidgetOnRemove?.apply(this, arguments);
          };
        }

        this.__tbgPanel = { container, label, iframe, widget, eventController };
        this.__tbgPanelInitialized = true;

        const refreshPanel = (output) => {
          tbgUpdateInfoPanel(this, output);
          tbgFitLegacyDomWidgetNode(this, panelNodeMinHeight);
          requestNodeRedraw(this);
        };

        this.onExecuted = function (output) {
          const r = safeApply(onExecuted, this, arguments);
          refreshPanel(output);
          return r;
        };

        this.onConfigure = function (info) {
          const r = safeApply(onConfigure, this, arguments);
          setTimeout(() => refreshPanel(info), 0);
          return r;
        };

        tbgFitLegacyDomWidgetNode(this, panelNodeMinHeight);
        setTimeout(() => refreshPanel(), 0);

        const prevOnRemoved = this.onRemoved;
        this.onRemoved = function () {
          try { this.__tbgPanel?.eventController?.abort(); } catch (_) {}
          this.__tbgPanel = null;
          this.__tbgPanelInitialized = false;
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
        container.style.minWidth = "0";
        container.style.maxWidth = "100%";
        container.style.overflow = "hidden";

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
        row.style.minWidth = "0";
        row.style.maxWidth = "100%";
        row.style.overflow = "hidden";

        // Seed field
        const seedField = document.createElement("input");
        seedField.type = "text";
        seedField.readOnly = true;
        seedField.style.flex = "1 1 auto";
        seedField.style.minWidth = "0";
        seedField.style.width = "0";
        seedField.style.boxSizing = "border-box";
        seedField.style.background = "#444";
        seedField.style.color = "#eee";
        seedField.style.border = "1px solid #666";
        seedField.style.borderRadius = "999px";
        seedField.style.padding = "4px 8px";
        seedField.style.fontSize = "12px";
        seedField.style.overflow = "hidden";
        seedField.style.textOverflow = "ellipsis";
        seedField.style.whiteSpace = "nowrap";

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
        copyBtn.style.flex = "0 0 auto";
        copyBtn.style.whiteSpace = "nowrap";
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
