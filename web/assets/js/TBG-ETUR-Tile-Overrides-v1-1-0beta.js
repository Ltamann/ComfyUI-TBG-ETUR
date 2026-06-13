/*******************************************************************************************
 *  Tile-Prompter – MULTI-INSTANCE version for TBG enhanced upscaler and refiner pro
 *
 * PURPOSE
 * ───────
 * Provide a "Tile Prompter" node that shows one row per image tile with:
 *  – prompt textarea
 *  – denoise value
 *  – seed
 *  – ControlNet strength
 *  – live tile preview
 *  – news and updates
 * Persist user edits to node.properties and a packed JSON string
 * Offer header-level bulk inputs (apply same prompt/denoise/seed/CNet to many rows)
 * Send status tags back from backend via custom event
 *
 * CHANGES vs your original
 * ────────────────────────
 * All runtime caches are now PER-NODE, keyed by node.id:
 *     getMsg(node) / getInp(node)
 * No shared global message/inputs arrays; multiple TilePrompters co-exist independently.
 * All references to window.TBG.TBGETUR.message / .inputs swapped to the per-node caches.
 * New Seed, Cnet entry's
 * Full reconstruction of oncreate and onexecuted and many other thinks.
 * Copyright (c) 2025/2026 Tobias Laarmann
 * Derivative of "McBoaty_v5.js" (c) 2024 David Asquiedge – MIT License with attribution.  Copyright (c) 2024 David Asquiedge
 *
 ******************************************************************************************/

import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";   // eslint-disable-line no-unused-vars
import { api } from "../../../scripts/api.js";
import {
    getNodeWidgets,
    isTBGNode,
    requestNodeRedraw,
    setNodeMinHeight,
} from "./TBG-ETUR-compat.js";

/*──────────────────────────────────────────────────────────────────────────*/
/* Backend connection status (unchanged; informational)                     */
/* Status is triggerd on page reload / refresh                              */
/*──────────────────────────────────────────────────────────────────────────*/

api.addEventListener("status", (event) => {
    if (!event.detail) {
        console.warn("[TBG_upscaler2] Status event fired with null detail");
        return;
    }
    if (event.detail.status === "disconnected") {
        console.warn("[TBG_upscaler2] Lost connection to server!");
    }
    if (event.detail.status === "connected") {
        console.log("[TBG_upscaler2] Reconnected to server.");
    }
});

api.addEventListener('reconnecting', () => {
  console.warn('[TBG] backend went down – waiting');
});

/*──────────────────────────────────────────────────────────────────────────*/
/* Global namespaces + per-node stores                                      */
/*──────────────────────────────────────────────────────────────────────────*/

if (!window.TBG) window.TBG = {};

if (!window.TBG.TBGETUR) {
    window.TBG.TBGETUR = {
        init:  false,
        clean: false,

        // Per-node stores
        messages: {},   // node.id -> {prompts, tiles, denoises, seeds, cnet_strength, cfg_overrides, model_overrides, cnetpipe_overrides, color_match_overrides}
        inputs:   {},   // node.id -> {prompts, tiles, denoises, seeds, cnet_strength, cfg_overrides, model_overrides, cnetpipe_overrides, color_match_overrides}

        // Header combo ranges
        params: {
            denoise:       { values: [] },
            seed:          { values: [] }, // reserved
            cnet_strength: { values: [] }  // reserved
        }
    };
}

/* Helpers to access per-node stores */

function ensureNodeStores(node) {
    const id = String(node?.id ?? "");
    if (!id) return null;
    const ns = window.TBG.TBGETUR;
    if (!ns.messages[id]) ns.messages[id] = { prompts: [], tiles: [], denoises: [], seeds: [], cnet_strength: [], cfg_overrides: [], model_overrides: [], cnetpipe_overrides: [], color_match_overrides: [], ignore_general_prompts: [] };
    if (!ns.inputs[id])   ns.inputs[id]   = { prompts: [], tiles: [], denoises: [], seeds: [], cnet_strength: [], cfg_overrides: [], model_overrides: [], cnetpipe_overrides: [], color_match_overrides: [], ignore_general_prompts: [] };
    for (const store of [ns.messages[id], ns.inputs[id]]) {
        if (!Array.isArray(store.cfg_overrides)) store.cfg_overrides = [];
        if (!Array.isArray(store.cnetpipe_overrides)) store.cnetpipe_overrides = [];
        if (!Array.isArray(store.color_match_overrides)) store.color_match_overrides = [];
        if (!Array.isArray(store.ignore_general_prompts)) store.ignore_general_prompts = [];
    }
    return id;
}

function getMsg(node) {
    const id = ensureNodeStores(node);
    return id ? window.TBG.TBGETUR.messages[id] : { prompts: [], tiles: [], denoises: [], seeds: [], cnet_strength: [], cfg_overrides: [], model_overrides: [], cnetpipe_overrides: [], color_match_overrides: [], ignore_general_prompts: [] };
}

function getInp(node) {
    const id = ensureNodeStores(node);
    return id ? window.TBG.TBGETUR.inputs[id] : { prompts: [], tiles: [], denoises: [], seeds: [], cnet_strength: [], cfg_overrides: [], model_overrides: [], cnetpipe_overrides: [], color_match_overrides: [], ignore_general_prompts: [] };
}

const TBG_TILE_OVERRIDES_NODE_NAMES = [
    "TBG ETUR Tile Overrides",
    "TBG_TilePrompter_v1",
];

const TBG_TILE_ROW_WIDGET_TYPE_LEGACY = "customtext";
const TBG_TILE_ROW_WIDGET_TYPE_MODERN = "tbg_tile_row";

function isTruthySetting(value) {
    if (value === true) return true;
    if (value === false || value == null) return false;
    const text = String(value).trim().toLowerCase();
    return ["true", "enabled", "enable", "on", "modern", "2.0", "nodes 2.0"].includes(text);
}

function getSettingValueSafe(key) {
    try {
        return app?.ui?.settings?.getSettingValue?.(key);
    } catch (_) {
        return undefined;
    }
}

function isModernNodeDesignEnabled() {
    const explicitKeys = [
        "Comfy.VueNodes.Enabled",
        "Comfy.ModernNodeDesign",
        "Comfy.Node.ModernDesign",
        "Comfy.Node.ModernNodeDesign",
        "Comfy.Node.UseModernDesign",
        "Comfy.Node.Nodes2",
        "Comfy.Node.Nodes2.0",
        "Comfy.UseNodes2",
        "Comfy.UseNodes2.0",
    ];
    for (const key of explicitKeys) {
        const value = getSettingValueSafe(key);
        if (value !== undefined) return isTruthySetting(value);
    }

    const settingDefs = app?.ui?.settings?.settings;
    const defEntries = settingDefs instanceof Map ? [...settingDefs.entries()] : Object.entries(settingDefs || {});
    for (const [key, def] of defEntries) {
        const haystack = [
            key,
            def?.id,
            def?.name,
            def?.label,
            def?.tooltip,
            def?.description,
        ].filter(Boolean).join(" ").toLowerCase();
        if (!haystack.includes("node")) continue;
        if (!/(modern|design|2\.0|nodes 2|nodes2)/.test(haystack)) continue;
        const id = def?.id || key;
        const value = getSettingValueSafe(id);
        if (value !== undefined) return isTruthySetting(value);
    }

    const values = app?.ui?.settings?.settingsValues || {};
    for (const [key, value] of Object.entries(values)) {
        const text = String(key).toLowerCase();
        if (!text.includes("node")) continue;
        if (!/(modern|design|2\.0|nodes2|new)/.test(text)) continue;
        return isTruthySetting(value);
    }

    const bodyClasses = String(document?.body?.className || "").toLowerCase();
    return bodyClasses.includes("modern") && bodyClasses.includes("node");
}

function getTileRowMode() {
    return isModernNodeDesignEnabled() ? "modern" : "legacy";
}

function getTileRowWidgetType() {
    return getTileRowMode() === "modern" ? TBG_TILE_ROW_WIDGET_TYPE_MODERN : TBG_TILE_ROW_WIDGET_TYPE_LEGACY;
}

function isTileOverridesNodeClass(nodeType, nodeData) {
    return isTBGNode(nodeType, nodeData, TBG_TILE_OVERRIDES_NODE_NAMES);
}

function getTileRowWidgetElement(widget) {
    return widget?.element || widget?.inputEl || null;
}

function isLegacyTileRowWidget(widget) {
    return widget?.type === "customtext" && /^tile_\d+$/i.test(widget?.name || "");
}

function isTBGTileRowWidget(widget) {
    const element = getTileRowWidgetElement(widget);
    if (!element?.querySelector) return false;
    if (![TBG_TILE_ROW_WIDGET_TYPE_LEGACY, TBG_TILE_ROW_WIDGET_TYPE_MODERN].includes(widget?.type) && !isLegacyTileRowWidget(widget)) return false;
    return !!element.querySelector('textarea[placeholder^="tile "]');
}

function getTileRowWidgetElements(node) {
    return getNodeWidgets(node)
        .filter(isTBGTileRowWidget)
        .map(getTileRowWidgetElement)
        .filter(Boolean);
}

function syncNodeWidgetsToTileState(node) {
    const msg = getMsg(node);
    const wrappers = getTileRowWidgetElements(node);

    wrappers.forEach((wrapperEl) => {
        const textarea = wrapperEl.querySelector('textarea[placeholder^="tile "]');
        if (!textarea) return;
        const idx = parseInt(textarea.placeholder.replace(/\D/g, ""), 10) - 1;
        const denoiseInput = wrapperEl.querySelector('input[placeholder^="denoise "]');
        const seedInput = wrapperEl.querySelector('input[placeholder^="seed "]');
        const cnetInput = wrapperEl.querySelector('input[placeholder^="cnet_strength "]');
        const cfgInput = wrapperEl.querySelector('input[placeholder^="cfg "]');
        const modelSelect = wrapperEl.querySelector('select[data-tbg-role="model_override"]');
        const cnetpipeSelect = wrapperEl.querySelector('select[data-tbg-role="cnetpipe_override"]');
        const colorMatchSelect = wrapperEl.querySelector('select[data-tbg-role="color_match_override"]');
        const ignoreGeneralCheckbox = wrapperEl.querySelector('input[data-tbg-role="ignore_general_prompt"]');

        msg.prompts[idx] = textarea.value || "";
        msg.denoises[idx] = denoiseInput?.value || "";
        msg.seeds[idx] = seedInput?.value || "";
        msg.cnet_strength[idx] = cnetInput?.value || "";
        msg.cfg_overrides[idx] = cfgInput?.value || "";
        msg.model_overrides[idx] = normalizeTileModelOverride(modelSelect?.value || "");
        msg.cnetpipe_overrides[idx] = normalizeTileCNetPipeOverride(cnetpipeSelect?.value || "");
        msg.color_match_overrides[idx] = normalizeTileColorMatchOverride(colorMatchSelect?.value || "");
        msg.ignore_general_prompts[idx] = !!ignoreGeneralCheckbox?.checked;

        setNodePropertySafe(node, `prompt_${idx}`, (msg.prompts[idx] || "").trim());
        setNodePropertySafe(node, `denoise_${idx}`, (msg.denoises[idx] || "").trim());
        setNodePropertySafe(node, `seed_${idx}`, (msg.seeds[idx] || "").trim());
        setNodePropertySafe(node, `cnet_strength_${idx}`, (msg.cnet_strength[idx] || "").trim());
        setNodePropertySafe(node, `cfg_override_${idx}`, (msg.cfg_overrides[idx] || "").trim());
        setNodePropertySafe(node, `model_override_${idx}`, (msg.model_overrides[idx] || "").trim());
        setNodePropertySafe(node, `cnetpipe_override_${idx}`, (msg.cnetpipe_overrides[idx] || "").trim());
        setNodePropertySafe(node, `color_match_override_${idx}`, (msg.color_match_overrides[idx] || "").trim());
        setNodePropertySafe(node, `ignore_general_prompt_${idx}`, msg.ignore_general_prompts[idx] ? "true" : "");
    });
}

function getCanvasElementForTBGEvents() {
    return app?.canvas?.canvas || document.querySelector("canvas");
}

function cloneEventForCanvas(event) {
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

function forwardTBGEventToCanvas(event) {
    const canvasEl = getCanvasElementForTBGEvents();
    if (!canvasEl) return false;
    event.preventDefault();
    event.stopPropagation();
    canvasEl.dispatchEvent(cloneEventForCanvas(event));
    return true;
}

function attachTBGCanvasInteractionForwarding(rootEl) {
    const controller = new AbortController();
    const signal = controller.signal;

    rootEl.style.touchAction = "none";

    rootEl.addEventListener("wheel", (event) => {
        forwardTBGEventToCanvas(event);
    }, { passive: false, signal });

    rootEl.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
    }, { signal });

    rootEl.addEventListener("pointerdown", (event) => {
        if (event.button === 1 || event.button === 2) {
            forwardTBGEventToCanvas(event);
        }
    }, { capture: true, signal });

    rootEl.addEventListener("pointermove", (event) => {
        if ((event.buttons & 4) === 4 || (event.buttons & 2) === 2) {
            forwardTBGEventToCanvas(event);
        }
    }, { signal });

    rootEl.addEventListener("pointerup", (event) => {
        if (event.button === 1 || event.button === 2) {
            forwardTBGEventToCanvas(event);
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

function getTileListFromExecutionMessage(message) {
    const candidates = [
        message?.tiles,
        message?.ui?.tiles,
        message?.output?.tiles,
        message?.value?.tiles,
        message?.value?.[0]?.tiles,
    ];
    return candidates.find((tiles) => Array.isArray(tiles)) || [];
}

function getTilerContextKeyFromExecutionMessage(message) {
    const candidates = [
        message?.tiler_context_key,
        message?.ui?.tiler_context_key,
        message?.output?.tiler_context_key,
        message?.value?.tiler_context_key,
        message?.value?.[0]?.tiler_context_key,
    ];
    return candidates.find((key) => key !== undefined && key !== null && String(key).trim() !== "") || "";
}

function getTileRowCount(msg) {
    if (Array.isArray(msg.tiles) && msg.tiles.length > 0) {
        return msg.tiles.length;
    }
    return Math.max(
        Array.isArray(msg.tiles) ? msg.tiles.length : 0,
        Array.isArray(msg.prompts) ? msg.prompts.length : 0,
        Array.isArray(msg.denoises) ? msg.denoises.length : 0,
        Array.isArray(msg.seeds) ? msg.seeds.length : 0,
        Array.isArray(msg.cnet_strength) ? msg.cnet_strength.length : 0,
        Array.isArray(msg.cfg_overrides) ? msg.cfg_overrides.length : 0,
        Array.isArray(msg.model_overrides) ? msg.model_overrides.length : 0,
        Array.isArray(msg.cnetpipe_overrides) ? msg.cnetpipe_overrides.length : 0,
        Array.isArray(msg.color_match_overrides) ? msg.color_match_overrides.length : 0,
        Array.isArray(msg.ignore_general_prompts) ? msg.ignore_general_prompts.length : 0
    );
}

const TBG_TILE_OVERRIDE_LEGACY_MIN_HEIGHT = 400;
const TBG_TILE_OVERRIDE_LEGACY_ROW_HEIGHT = 186;
const TBG_TILE_OVERRIDE_LEGACY_HEADER_PAD = 80;
const TBG_TILE_OVERRIDE_LEGACY_WIDGET_HEIGHT = 30;
const TBG_TILE_OVERRIDE_LEGACY_MAX_HEIGHT = 20000;

function getWidgetHeightSafe(widget, fallback) {
    try {
        const height = widget?.getHeight?.();
        if (Number.isFinite(height) && height > 0) return height;
    } catch (_) {}
    try {
        const height = widget?.getMinHeight?.();
        if (Number.isFinite(height) && height > 0) return height;
    } catch (_) {}
    return fallback;
}

function getTileOverrideRequiredLegacyHeight(node) {
    const msg = getMsg(node);
    const tileCount = getTileRowCount(msg);
    const widgets = getNodeWidgets(node);
    let tileWidgetHeight = 0;
    let visibleHeaderWidgets = 0;

    for (const widget of widgets) {
        if (!widget || widget.hidden) continue;
        if (isTBGTileRowWidget(widget) || isLegacyTileRowWidget(widget)) {
            tileWidgetHeight += getWidgetHeightSafe(widget, TBG_TILE_OVERRIDE_LEGACY_ROW_HEIGHT);
        } else {
            visibleHeaderWidgets += 1;
        }
    }

    const rowHeight = Math.max(tileWidgetHeight, tileCount * TBG_TILE_OVERRIDE_LEGACY_ROW_HEIGHT);
    const headerHeight = TBG_TILE_OVERRIDE_LEGACY_HEADER_PAD +
        visibleHeaderWidgets * TBG_TILE_OVERRIDE_LEGACY_WIDGET_HEIGHT;

    let computedHeight = 0;
    try {
        const computed = typeof node?.computeSize === "function" ? node.computeSize() : null;
        computedHeight = Array.isArray(computed) ? Number(computed[1] || 0) : 0;
    } catch (_) {}

    const required = Math.max(
        TBG_TILE_OVERRIDE_LEGACY_MIN_HEIGHT,
        computedHeight,
        headerHeight + rowHeight
    );
    return Math.min(TBG_TILE_OVERRIDE_LEGACY_MAX_HEIGHT, required);
}

function fitTileOverrideNodeHeight(node, { allowShrink = false } = {}) {
    if (getTileRowMode() !== "legacy") {
        requestNodeRedraw(node);
        return false;
    }

    const requiredHeight = getTileOverrideRequiredLegacyHeight(node);
    if (allowShrink) {
        const width = Array.isArray(node?.size) ? node.size[0] : node?.size?.[0];
        const currentHeight = Array.isArray(node?.size) ? node.size[1] : node?.size?.[1];
        if (Number(currentHeight || 0) > requiredHeight) {
            if (typeof node?.setSize === "function") {
                node.setSize([Number(width ?? 250), requiredHeight]);
            } else if (Array.isArray(node?.size)) {
                node.size[1] = requiredHeight;
            } else if (node?.size && typeof node.size === "object") {
                node.size[1] = requiredHeight;
            }
            requestNodeRedraw(node);
            return true;
        }
    }

    setNodeMinHeight(node, requiredHeight);
    return true;
}

function scheduleTileOverrideNodeHeightFit(node, options = {}) {
    const didFit = fitTileOverrideNodeHeight(node, options);
    if (!didFit) return;

    const refit = () => fitTileOverrideNodeHeight(node, options);
    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(refit);
    } else {
        setTimeout(refit, 0);
    }
}

function hasTileImage(tile) {
    return !!(tile && tile.filename && tile.type);
}

function hydrateTileOverrideNode(node) {
    syncPropertiesToMessage(node);
    const msg = getMsg(node);
    if (!Array.isArray(msg.tiles)) msg.tiles = [];
    return msg;
}

function refreshTileOverrideNode(node, { pack = false } = {}) {
    hydrateTileOverrideNode(node);
    if (!Array.isArray(node?.widgets)) return;
    TBGUpscalerTileNodeWidget.refresh(node);
    scheduleTileOverrideNodeHeightFit(node, { allowShrink: true });
    if (pack) {
        packEditsToJson(node);
    }
}

function isTileOverrideRuntimeNode(node) {
    const nodeType = String(node?.type ?? node?.constructor?.comfyClass ?? node?.title ?? "");
    return nodeType === "TBG_TilePrompter_v1" || nodeType === "TBG ETUR Tile Overrides";
}

function refreshAllTileOverrideNodes({ pack = false } = {}) {
    if (!app?.graph?._nodes) return;
    for (const node of app.graph._nodes) {
        if (!isTileOverrideRuntimeNode(node)) continue;
        refreshTileOverrideNode(node, { pack });
    }
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Property key builders + safe property access                             */
/*──────────────────────────────────────────────────────────────────────────*/

function propKeyPrompt(i) {return `prompt_${i}`;}
function propKeyDenoise(i) {return `denoise_${i}`;}
function propKeySeed(i) {return `seed_${i}`;}
function propKeyCNet(i) {return `cnet_strength_${i}`;}
function propKeyCFG(i) {return `cfg_override_${i}`;}
function propKeyModel(i) {return `model_override_${i}`;}
function propKeyCNetPipe(i) {return `cnetpipe_override_${i}`;}
function propKeyColorMatch(i) {return `color_match_override_${i}`;}
function propKeyIgnoreGeneralPrompt(i) {return `ignore_general_prompt_${i}`;}

const TBG_TILE_MODEL_OPTIONS = ["normal", "model 1", "model 2", "model 3"];
const TBG_TILE_CNETPIPE_OPTIONS = ["normal", "cnetpipe 1", "cnetpipe 2", "cnetpipe 3"];
const TBG_TILE_COLOR_MATCH_OPTIONS = [
    "as_refiner",
    "protect_new_generated_content",
    "color_match_from_origin",
    "color_match_off",
];
const TBG_TILE_MODEL_LABELS = {
    normal: "Preset Model",
    "model 1": "model 1",
    "model 2": "model 2",
    "model 3": "model 3",
};
const TBG_TILE_CNETPIPE_LABELS = {
    normal: "Preset Cnet",
    "cnetpipe 1": "cnetpipe 1",
    "cnetpipe 2": "cnetpipe 2",
    "cnetpipe 3": "cnetpipe 3",
};
const TBG_TILE_COLOR_MATCH_LABELS = {
    as_refiner: "Preset Color Match",
    protect_new_generated_content: "Protect New Generated Content",
    color_match_from_origin: "Color Match From Origin",
    color_match_off: "Color Match Off",
};

function normalizeTileModelOverride(value) {
    const raw = String(value ?? "").trim().toLowerCase().replace(/[_-]+/g, " ");
    if (raw === "" || raw === "normal" || raw === "default" || raw === "none") return "";
    if (raw === "1" || raw === "model1") return "model 1";
    if (raw === "2" || raw === "model2") return "model 2";
    if (raw === "3" || raw === "model3") return "model 3";
    return TBG_TILE_MODEL_OPTIONS.includes(raw) ? raw : "";
}

function normalizeTileCNetPipeOverride(value) {
    const raw = String(value ?? "").trim().toLowerCase().replace(/[_-]+/g, " ");
    if (raw === "" || raw === "normal" || raw === "default" || raw === "none") return "";
    if (raw === "1" || raw === "cnetpipe1" || raw === "cnet pipe 1" || raw === "controlnet pipe 1") return "cnetpipe 1";
    if (raw === "2" || raw === "cnetpipe2" || raw === "cnet pipe 2" || raw === "controlnet pipe 2") return "cnetpipe 2";
    if (raw === "3" || raw === "cnetpipe3" || raw === "cnet pipe 3" || raw === "controlnet pipe 3") return "cnetpipe 3";
    return TBG_TILE_CNETPIPE_OPTIONS.includes(raw) ? raw : "";
}

function normalizeTileColorMatchOverride(value) {
    const raw = String(value ?? "").trim().toLowerCase().replace(/[_\s-]+/g, " ");
    if (raw === "" || raw === "normal" || raw === "default" || raw === "none" || raw === "as refiner" || raw === "cm as refiner") return "";
    if (raw === "color match off" || raw === "color mach off" || raw === "cm off" || raw === "off") return "color_match_off";
    if (raw === "color_match_off") return "color_match_off";
    if (raw === "protect generated" || raw === "protect new generated content" || raw === "protect generated content") return "protect_new_generated_content";
    if (raw === "protect_new_generated_content") return "protect_new_generated_content";
    if (raw === "color match from origin" || raw === "match from origin" || raw === "from origin" || raw === "origin") return "color_match_from_origin";
    if (raw === "color_match_from_origin") return "color_match_from_origin";
    if (raw === "full match" || raw === "full_match") return "color_match_from_origin";
    return "";
}

function colorMatchUiValue(value) {
    return normalizeTileColorMatchOverride(value) || "as_refiner";
}

function normalizeTileIgnoreGeneralPrompt(value) {
    if (value === true) return true;
    if (value === false || value == null) return false;
    const raw = String(value).trim().toLowerCase();
    return ["1", "true", "yes", "on", "checked", "enabled"].includes(raw);
}

function styleTileOverrideSelect(selectEl, height = "36px") {
    if (!selectEl) return;
    selectEl.style.opacity = 0.6;
    selectEl.style.height = height;
    selectEl.style.maxWidth = "100%";
    selectEl.style.boxSizing = "border-box";
    selectEl.style.flexShrink = "0";
    selectEl.style.appearance = "none";
    selectEl.style.webkitAppearance = "none";
    selectEl.style.mozAppearance = "none";
    selectEl.style.backgroundImage = "none";
    selectEl.style.backgroundColor = "var(--comfy-input-bg, #222)";
    selectEl.style.color = "var(--input-text, var(--fg-color, #ddd))";
    selectEl.style.borderColor = "var(--border-color, #444)";
    selectEl.style.paddingLeft = "8px";
    selectEl.style.paddingRight = "8px";
}

function setNodePropertySafe(node, key, value) {
    try {
        node.setProperty(key, value);
        node.setDirtyCanvas(true);
    } catch (_) {/* ignore */ }
}

function getNodeProperty(node, key, fallback = "") {
    try {
        const v = node.properties?.[key];
        return (v === undefined || v === null) ? fallback : v;
    } catch (_) {
        return fallback;
    }
}

function getPackedTileEdits(node) {
    const raw = getNodeProperty(node, "tile_edits_json", "");
    if (typeof raw !== "string" || raw.trim() === "") return {};
    try {
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") return {};
        delete parsed.tiles;
        return parsed;
    } catch (e) {
        console.warn("[TBG] tile_edits_json parse failed:", e);
        return {};
    }
}

function imageDataToUrl(data) {
  if (!hasTileImage(data)) return "";
  return api.apiURL(
    `/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`
  );
}

function getTileMaskData(tile) {
    return hasTileImage(tile?.mask) ? tile.mask : null;
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Small utilities                                                           */
/*──────────────────────────────────────────────────────────────────────────*/

function debounce(fn, delay = 1000) {
    let t;
    return (...args) => {
        clearTimeout(t);
        t = setTimeout(() => fn(...args), delay);
    };
}
// ONE requeue per execution session
const requeueSession = new Map();
function bumpRequeue(node) {
  const id = String(node.id);
  const now = Date.now();

  if (!requeueSession.has(id)) {
    requeueSession.set(id, { lastRequeue: 0, sessionStart: now });
  }

  const session = requeueSession.get(id);

  // Reset session after 5s idle (prevents stale locks)
  if (now - session.sessionStart > 5000) {
    session.lastRequeue = 0;
    session.sessionStart = now;
  }

  // Only requeue if 500ms passed since last
  if (now - session.lastRequeue > 500) {
    session.lastRequeue = now;

    const widget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
    TBGUpscalerTileNodeWidget.setValue(node, 'requeue', (widget?.value ?? 0) + 1);

    console.log(`ðŸ”¥ REQUEUE ${widget?.value + 1} for node ${id}`);
    return true;
  }

  return false; // Already queued
}


function clearExcessTileProperties(node, currentTileCount) {
    const MAX_CLEAR = 512;
    for (let i = currentTileCount; i < MAX_CLEAR; i++) {
        const hasAnyProperty = [
            getNodeProperty(node, `prompt_${i}`, null),
            getNodeProperty(node, `denoise_${i}`, null),
            getNodeProperty(node, `seed_${i}`, null),
            getNodeProperty(node, `cnet_strength_${i}`, null),
            getNodeProperty(node, `cfg_override_${i}`, null),
            getNodeProperty(node, `model_override_${i}`, null),
            getNodeProperty(node, `cnetpipe_override_${i}`, null),
            getNodeProperty(node, `color_match_override_${i}`, null),
            getNodeProperty(node, `ignore_general_prompt_${i}`, null)
        ].some(prop => prop !== null);
        if (!hasAnyProperty) continue;
        try {
            delete node.properties[propKeyPrompt(i)];
            delete node.properties[propKeyDenoise(i)];
            delete node.properties[propKeySeed(i)];
            delete node.properties[propKeyCNet(i)];
            delete node.properties[propKeyCFG(i)];
            delete node.properties[propKeyModel(i)];
            delete node.properties[propKeyCNetPipe(i)];
            delete node.properties[propKeyColorMatch(i)];
            delete node.properties[propKeyIgnoreGeneralPrompt(i)];
        } catch (_) {}
    }
}

function trimTileStateToCount(node, count) {
    const safeCount = Math.max(0, Number(count) || 0);
    const msg = getMsg(node);
    const inp = getInp(node);
    for (const store of [msg, inp]) {
        store.prompts = (store.prompts || []).slice(0, safeCount);
        store.denoises = (store.denoises || []).slice(0, safeCount);
        store.seeds = (store.seeds || []).slice(0, safeCount);
        store.cnet_strength = (store.cnet_strength || []).slice(0, safeCount);
        store.cfg_overrides = (store.cfg_overrides || []).slice(0, safeCount);
        store.model_overrides = (store.model_overrides || []).slice(0, safeCount);
        store.cnetpipe_overrides = (store.cnetpipe_overrides || []).slice(0, safeCount);
        store.color_match_overrides = (store.color_match_overrides || []).slice(0, safeCount);
        store.ignore_general_prompts = (store.ignore_general_prompts || []).slice(0, safeCount);
        if (Array.isArray(store.tiles)) store.tiles = store.tiles.slice(0, safeCount);
    }
    clearExcessTileProperties(node, safeCount);
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Per-node: properties â†’ message caches                                     */
/*──────────────────────────────────────────────────────────────────────────*/

function syncPropertiesToMessage(node) {
    const msg = getMsg(node);
    const packed = getPackedTileEdits(node);
    const prompts = Array.isArray(packed.prompts) ? packed.prompts.slice() : [];
    const denoises = Array.isArray(packed.denoises) ? packed.denoises.slice() : [];
    const seeds = Array.isArray(packed.seeds) ? packed.seeds.slice() : [];
    const cnet = Array.isArray(packed.cnet_strength) ? packed.cnet_strength.slice() : [];
    const cfgOverrides = Array.isArray(packed.cfg_overrides) ? packed.cfg_overrides.slice() : [];
    const modelOverrides = Array.isArray(packed.model_overrides) ? packed.model_overrides.slice() : [];
    const cnetpipeOverrides = Array.isArray(packed.cnetpipe_overrides) ? packed.cnetpipe_overrides.slice() : [];
    const colorMatchOverrides = Array.isArray(packed.color_match_overrides) ? packed.color_match_overrides.slice() : [];
    const ignoreGeneralPrompts = Array.isArray(packed.ignore_general_prompts) ? packed.ignore_general_prompts.slice().map(normalizeTileIgnoreGeneralPrompt) : [];
    if (packed._tbg_tiler_context_key) {
        msg.tiler_context_key = String(packed._tbg_tiler_context_key);
    }

    const MAX = 512;
    let actualTileCount = Math.max(
        Array.isArray(msg.tiles) ? msg.tiles.length : 0,
        prompts.length,
        denoises.length,
        seeds.length,
        cnet.length,
        cfgOverrides.length,
        modelOverrides.length,
        cnetpipeOverrides.length,
        colorMatchOverrides.length,
        ignoreGeneralPrompts.length
    );

    for (let i = 0; i < MAX; i++) {
        const p = getNodeProperty(node, `prompt_${i}`, null);
        const d = getNodeProperty(node, `denoise_${i}`, null);
        const s = getNodeProperty(node, `seed_${i}`, null);
        const c = getNodeProperty(node, `cnet_strength_${i}`, null);
        const cfg = getNodeProperty(node, `cfg_override_${i}`, null);
        const m = getNodeProperty(node, `model_override_${i}`, null);
        const cp = getNodeProperty(node, `cnetpipe_override_${i}`, null);
        const cm = getNodeProperty(node, `color_match_override_${i}`, null);
        const ig = getNodeProperty(node, `ignore_general_prompt_${i}`, null);

        if (p !== null || d !== null || s !== null || c !== null || cfg !== null || m !== null || cp !== null || cm !== null || ig !== null) {
            actualTileCount = i + 1;
            if (p !== null) prompts[i] = p;
            else prompts[i] = prompts[i] ?? "";
            if (d !== null) denoises[i] = d;
            else denoises[i] = denoises[i] ?? "";
            if (s !== null) seeds[i] = s;
            else seeds[i] = seeds[i] ?? "";
            if (c !== null) cnet[i] = c;
            else cnet[i] = cnet[i] ?? "";
            if (cfg !== null) cfgOverrides[i] = cfg;
            else cfgOverrides[i] = cfgOverrides[i] ?? "";
            if (m !== null) modelOverrides[i] = normalizeTileModelOverride(m);
            else modelOverrides[i] = normalizeTileModelOverride(modelOverrides[i] ?? "");
            if (cp !== null) cnetpipeOverrides[i] = normalizeTileCNetPipeOverride(cp);
            else cnetpipeOverrides[i] = normalizeTileCNetPipeOverride(cnetpipeOverrides[i] ?? "");
            if (cm !== null) colorMatchOverrides[i] = normalizeTileColorMatchOverride(cm);
            else colorMatchOverrides[i] = normalizeTileColorMatchOverride(colorMatchOverrides[i] ?? "");
            if (ig !== null) ignoreGeneralPrompts[i] = normalizeTileIgnoreGeneralPrompt(ig);
            else ignoreGeneralPrompts[i] = normalizeTileIgnoreGeneralPrompt(ignoreGeneralPrompts[i] ?? false);
        }
    }

    if (actualTileCount > 0) {
        prompts.length = actualTileCount;
        denoises.length = actualTileCount;
        seeds.length = actualTileCount;
        cnet.length = actualTileCount;
        cfgOverrides.length = actualTileCount;
        modelOverrides.length = actualTileCount;
        cnetpipeOverrides.length = actualTileCount;
        colorMatchOverrides.length = actualTileCount;
        ignoreGeneralPrompts.length = actualTileCount;
    }

    msg.prompts = prompts;
    msg.denoises = denoises;
    msg.seeds = seeds;
    msg.cnet_strength = cnet;
    msg.cfg_overrides = cfgOverrides;
    msg.model_overrides = modelOverrides.map(normalizeTileModelOverride);
    msg.cnetpipe_overrides = cnetpipeOverrides.map(normalizeTileCNetPipeOverride);
    msg.color_match_overrides = colorMatchOverrides.map(normalizeTileColorMatchOverride);
    msg.ignore_general_prompts = ignoreGeneralPrompts.map(normalizeTileIgnoreGeneralPrompt);

    if (actualTileCount > 0 && (!Array.isArray(msg.tiles) || msg.tiles.length < actualTileCount)) {
        const existingTiles = Array.isArray(msg.tiles) ? msg.tiles : [];
        msg.tiles = Array.from({ length: actualTileCount }, (_, i) => {
            return existingTiles[i] || { filename: "", type: "", subfolder: "" };
        });
    }

    if (actualTileCount > 0) {
        clearExcessTileProperties(node, actualTileCount);
    }
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Persist wrappers â†’ properties â†’ per-node caches                           */
/*──────────────────────────────────────────────────────────────────────────*/

function syncAllWrappersToProperties(node) {
    const wrappers = getTileRowWidgetElements(node);

    const msg = getMsg(node);

    wrappers.forEach(wrapperEl => {
        const textarea = wrapperEl.querySelector('textarea[placeholder^="tile "]');
        if (!textarea) return;

        const idx = parseInt(textarea.placeholder.replace(/\D/g, ""), 10) - 1;

        const denoiseInput = wrapperEl.querySelector('input[placeholder^="denoise "]');
        const seedInput = wrapperEl.querySelector('input[placeholder^="seed "]');
        const cnetInput = wrapperEl.querySelector('input[placeholder^="cnet_strength "]');
        const cfgInput = wrapperEl.querySelector('input[placeholder^="cfg "]');
        const modelSelect = wrapperEl.querySelector('select[data-tbg-role="model_override"]');
        const cnetpipeSelect = wrapperEl.querySelector('select[data-tbg-role="cnetpipe_override"]');

        const rawD = (denoiseInput?.value || "").trim();
        const rawS = (seedInput?.value || "").trim();
        const rawC = (cnetInput?.value || "").trim();
        const rawCFG = (cfgInput?.value || "").trim();
        const normM = normalizeTileModelOverride(modelSelect?.value || "");
        const normCP = normalizeTileCNetPipeOverride(cnetpipeSelect?.value || "");
        const normCM = normalizeTileColorMatchOverride(colorMatchSelect?.value || "");
        const normIG = !!ignoreGeneralCheckbox?.checked;

        const normD = rawD === '' ? '' :
            (window.TBG.TBGWidgets.validateDenoiseValue(rawD) ? window.TBG.TBGWidgets.formatDenoiseValue(rawD) : '');
        const normS = rawS === '' ? '' :
            (window.TBG.TBGWidgets.validateSeedValue(rawS) ? window.TBG.TBGWidgets.formatSeedValue(rawS) : '');
        const normC = rawC === '' ? '' :
            (window.TBG.TBGWidgets.validateDenoiseValue(rawC) ? window.TBG.TBGWidgets.formatDenoiseValue(rawC) : '');
        const normCFG = rawCFG === '' ? '' :
            (window.TBG.TBGWidgets.validateCFGValue(rawCFG) ? window.TBG.TBGWidgets.formatCFGValue(rawCFG) : '');

        setNodePropertySafe(node, propKeyDenoise(idx), normD);
        setNodePropertySafe(node, propKeySeed(idx), normS);
        setNodePropertySafe(node, propKeyCNet(idx), normC);
        setNodePropertySafe(node, propKeyCFG(idx), normCFG);
        setNodePropertySafe(node, propKeyModel(idx), normM);
        setNodePropertySafe(node, propKeyCNetPipe(idx), normCP);
        setNodePropertySafe(node, propKeyColorMatch(idx), normCM);
        setNodePropertySafe(node, propKeyIgnoreGeneralPrompt(idx), normIG ? "true" : "");

        msg.denoises[idx] = normD;
        msg.seeds[idx] = normS;
        msg.cnet_strength[idx] = normC;
        msg.cfg_overrides[idx] = normCFG;
        msg.model_overrides[idx] = normM;
        msg.cnetpipe_overrides[idx] = normCP;
        msg.color_match_overrides[idx] = normCM;
        msg.ignore_general_prompts[idx] = normIG;

        if (denoiseInput) denoiseInput.value = normD;
        if (seedInput) seedInput.value = normS;
        if (cnetInput) cnetInput.value = normC;
        if (cfgInput) cfgInput.value = normCFG;
        if (modelSelect) modelSelect.value = normM || "normal";
        if (cnetpipeSelect) cnetpipeSelect.value = normCP || "normal";
        if (colorMatchSelect) colorMatchSelect.value = colorMatchUiValue(normCM);
        if (ignoreGeneralCheckbox) ignoreGeneralCheckbox.checked = normIG;
    });
    console.log('[TBG] JSON syncAllWrappersToProperties')
    packEditsToJson(node);
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Build + mirror packed JSON to node + optional POST to backend             */
/*──────────────────────────────────────────────────────────────────────────*/

async function packEditsToJson(node, force = true) {

    try {
        if (!node) {
            console.warn("packEditsToJson called without a node");
            return false;
        }
        const msg = getMsg(node);

        const tileCount = Array.isArray(msg.tiles) && msg.tiles.length > 0 ? msg.tiles.length : getTileRowCount(msg);
        const prompts = (msg.prompts || []).slice(0, tileCount);
        const denoises = (msg.denoises || []).slice(0, tileCount);
        const seeds = (msg.seeds || []).slice(0, tileCount);
        const cnet = (msg.cnet_strength || []).slice(0, tileCount);
        const cfgOverrides = (msg.cfg_overrides || []).slice(0, tileCount);
        const modelOverrides = (msg.model_overrides || []).slice(0, tileCount).map(normalizeTileModelOverride);
        const cnetpipeOverrides = (msg.cnetpipe_overrides || []).slice(0, tileCount).map(normalizeTileCNetPipeOverride);
        const colorMatchOverrides = (msg.color_match_overrides || []).slice(0, tileCount).map(normalizeTileColorMatchOverride);
        const ignoreGeneralPrompts = (msg.ignore_general_prompts || []).slice(0, tileCount).map(normalizeTileIgnoreGeneralPrompt);

        const normDenoises = denoises.map(v => {
            const s = String(v ?? '').trim();
            if (s === '') return '';
            return window.TBG.TBGWidgets.validateDenoiseValue(s) ?
                window.TBG.TBGWidgets.formatDenoiseValue(s) :
                '';
        });
        const normSeeds = seeds.map(v => {
            const s = String(v ?? '').trim();
            if (s === '') return '';
            return window.TBG.TBGWidgets.validateSeedValue(s) ?
                window.TBG.TBGWidgets.formatSeedValue(s) :
                '';
        });
        const normCnet = cnet.map(v => {
            const s = String(v ?? '').trim();
            if (s === '') return '';
            return window.TBG.TBGWidgets.validateDenoiseValue(s) ?
                window.TBG.TBGWidgets.formatDenoiseValue(s) :
                '';
        });
        const normCFG = cfgOverrides.map(v => {
            const s = String(v ?? '').trim();
            if (s === '') return '';
            return window.TBG.TBGWidgets.validateCFGValue(s) ?
                window.TBG.TBGWidgets.formatCFGValue(s) :
                '';
        });

        msg.denoises = normDenoises;
        msg.seeds = normSeeds;
        msg.cnet_strength = normCnet;
        msg.cfg_overrides = normCFG;
        msg.model_overrides = modelOverrides;
        msg.cnetpipe_overrides = cnetpipeOverrides;
        msg.color_match_overrides = colorMatchOverrides;
        msg.ignore_general_prompts = ignoreGeneralPrompts;
        msg.prompts = prompts;

        const hasContent = [...prompts, ...normDenoises, ...normSeeds, ...normCnet, ...normCFG, ...modelOverrides, ...cnetpipeOverrides, ...colorMatchOverrides]
            .some(v => v != null && v !== '');
        const hasCheckedIgnoreGeneral = ignoreGeneralPrompts.some(Boolean);

        const packed = JSON.stringify({
            prompts,
            denoises: normDenoises,
            seeds: normSeeds,
            cnet_strength: normCnet,
            cfg_overrides: normCFG,
            model_overrides: modelOverrides,
            cnetpipe_overrides: cnetpipeOverrides,
            color_match_overrides: colorMatchOverrides,
            ignore_general_prompts: ignoreGeneralPrompts,
            _tbg_tiler_context_key: msg.tiler_context_key || "",
        });
        console.log('[TBG] JSON',packed)
        setNodePropertySafe(node, "tile_edits_json", packed);

        if (hasContent || hasCheckedIgnoreGeneral) {
            try {
                await fetch(api.apiURL("/TBG/set_tile_edits_json"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        node: String(node.id),
                        tile_edits_json: packed
                    }),
                });
            } catch (e) {
                console.warn("mirror tile_edits_json failed:", e);
            }
        } else if (force) {
            try {
                await fetch(api.apiURL("/TBG/set_tile_edits_json"), {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        node: String(node.id),
                        tile_edits_json: ""
                    }),
                });
            } catch (e) {
                console.warn("clear tile_edits_json mirror failed:", e);
            }
        }
        return true;
    } catch (e) {
        console.warn("packEditsToJson failed:", e);
        return false;
    }
}

/*──────────────────────────────────────────────────────────────────────────*/
/* UI Widgets and validators                                                 */
/*──────────────────────────────────────────────────────────────────────────*/

window.TBG.TBGWidgets = {
    validateDenoiseValue: (value) => {
        if (value === '') return true;
        const num = parseFloat(value);
        return !isNaN(num) && num >= 0.00 && num <= 1.00;
    },
    formatDenoiseValue: (value) => {
        if (value === '') return value;
        let strValue = String(value).trim();
        if (strValue.startsWith('.')) strValue = '0' + strValue;
        return parseFloat(strValue).toFixed(2);
    },
    validateCFGValue: (value) => {
        if (value === '') return true;
        const num = parseFloat(value);
        return !isNaN(num) && num >= -10.00 && num <= 100.00;
    },
    formatCFGValue: (value) => {
        if (value === '') return value;
        let strValue = String(value).trim();
        if (strValue.startsWith('.')) strValue = '0' + strValue;
        return parseFloat(strValue).toFixed(2);
    },
    validateSeedValue: (_value) => {
        return true; // sanitization-only approach
    },
    formatSeedValue: (value) => {
        let s = String(value ?? '').trim();
        const isNeg = s.startsWith('-');
        if (isNeg) s = s.slice(1);
        s = s.replace(/\D+/g, '');
        if (isNeg && s.length > 0) s = '-' + s;
        return s;
    },

    WRAPPER: (key, index, prompt, tile, denoise, seed, cnet_strength, cfgOverride, modelOverride, cnetpipeOverride, colorMatchOverride, node) => {
        const rowMode = getTileRowMode();
        const inputEl = document.createElement("div");
        inputEl.className = `comfy-wrapper-tgb tbg-row-${rowMode}`;

        const wrapper = document.createElement("div");
        wrapper.style.height = "100%";
        wrapper.style.display = "flex";
        wrapper.style.alignItems = "center";
        wrapper.style.gap = "10px";

        const text = document.createElement("p");
        text.textContent = String(index + 1).padStart(2, '0');

        const seedInput = document.createElement("input");
        seedInput.style.opacity = 0.6;
        seedInput.style.height = "100%";
        seedInput.style.maxWidth = "3.5rem";
        seedInput.style.flexShrink = "0";
        seedInput.className = "comfy-multiline-input";
        seedInput.value = seed || '';
        seedInput.placeholder = "seed " + text.textContent;
        seedInput.dataId = "tile " + index;
        seedInput.dataNodeId = node.id;

          const persistSeed = () => {
            const msg = getMsg(node);
            let v = seedInput.value.trim();
            if (!window.TBG.TBGWidgets.validateSeedValue(v)) v = '';
            else v = window.TBG.TBGWidgets.formatSeedValue(v);
            if (seedInput.value !== v) seedInput.value = v;
            if (msg.seeds[index] !== v) {  // âœ… FIXED: msg.seeds
              msg.seeds[index] = v;
              setNodePropertySafe(node, propKeySeed(index), v);  // âœ… FIXED: propKeySeed
            if (bumpRequeue(node)) {
                  console.log('[TBG] JSON bumpRequeue seeds')
                  packEditsToJson(node);  // Only once per session
                }
            }
          };

        seedInput.addEventListener('input', debounce(persistSeed, 2000));

        seedInput.addEventListener('change', persistSeed); // Immediate save on field exit




        const cnet_strengthInput = document.createElement("input");
        cnet_strengthInput.style.opacity = 0.6;
        cnet_strengthInput.style.height = "40px";
        cnet_strengthInput.style.maxWidth = "100%";
        cnet_strengthInput.style.boxSizing = "border-box";
        cnet_strengthInput.style.flexShrink = "0";
        cnet_strengthInput.className = "comfy-multiline-input";
        cnet_strengthInput.value = (cnet_strength ?? '');
        cnet_strengthInput.placeholder = "cnet_strength " + text.textContent;
        cnet_strengthInput.dataId = "tile " + index;
        cnet_strengthInput.dataNodeId = node.id;

          const persistCnet = () => {
                const msg = getMsg(node);
                let v = cnet_strengthInput.value.trim();
                if (!window.TBG.TBGWidgets.validateDenoiseValue(v)) v = '';
                else v = window.TBG.TBGWidgets.formatDenoiseValue(v);
                if (cnet_strengthInput.value !== v) cnet_strengthInput.value = v;
                if (msg.cnet_strength[index] !== v) {
                  msg.cnet_strength[index] = v;
                  setNodePropertySafe(node, propKeyCNet(index), v);
                  if (bumpRequeue(node)) {
                  console.log('[TBG] JSON bumpRequeue cnet_strength')
                  packEditsToJson(node);  // Only once per session
                }
                }
              };
        cnet_strengthInput.addEventListener('input', debounce(persistCnet, 2000));

        cnet_strengthInput.addEventListener('change', persistCnet); // Immediate save on field exit

        const cfgInput = document.createElement("input");
        cfgInput.style.opacity = 0.6;
        cfgInput.style.height = "40px";
        cfgInput.style.maxWidth = "100%";
        cfgInput.style.boxSizing = "border-box";
        cfgInput.style.flexShrink = "0";
        cfgInput.className = "comfy-multiline-input";
        cfgInput.value = (cfgOverride ?? '');
        cfgInput.placeholder = "cfg " + text.textContent;
        cfgInput.dataId = "tile " + index;
        cfgInput.dataNodeId = node.id;

        const persistCFG = () => {
            const msg = getMsg(node);
            let v = cfgInput.value.trim();
            if (!window.TBG.TBGWidgets.validateCFGValue(v)) v = '';
            else v = window.TBG.TBGWidgets.formatCFGValue(v);
            if (cfgInput.value !== v) cfgInput.value = v;
            if (msg.cfg_overrides[index] !== v) {
                msg.cfg_overrides[index] = v;
                setNodePropertySafe(node, propKeyCFG(index), v);
                if (bumpRequeue(node)) {
                    console.log('[TBG] JSON bumpRequeue cfg_overrides')
                    packEditsToJson(node);
                }
            }
        };
        cfgInput.addEventListener('input', debounce(persistCFG, 2000));
        cfgInput.addEventListener('change', persistCFG);


        const modelSelect = document.createElement("select");
        modelSelect.className = "comfy-multiline-input";
        styleTileOverrideSelect(modelSelect, "36px");
        modelSelect.placeholder = "model " + text.textContent;
        modelSelect.dataId = "tile " + index;
        modelSelect.dataNodeId = node.id;
        modelSelect.dataset.tbgRole = "model_override";
        const currentModelOverride = normalizeTileModelOverride(modelOverride);
        for (const optionValue of TBG_TILE_MODEL_OPTIONS) {
            const option = document.createElement("option");
            option.value = optionValue;
            option.textContent = TBG_TILE_MODEL_LABELS[optionValue] || optionValue;
            modelSelect.appendChild(option);
        }
        modelSelect.value = currentModelOverride || "normal";

        const persistModelOverride = () => {
            const msg = getMsg(node);
            const v = normalizeTileModelOverride(modelSelect.value);
            const uiValue = v || "normal";
            if (modelSelect.value !== uiValue) modelSelect.value = uiValue;
            if (msg.model_overrides[index] !== v) {
                msg.model_overrides[index] = v;
                setNodePropertySafe(node, propKeyModel(index), v);
                if (bumpRequeue(node)) {
                    console.log('[TBG] JSON bumpRequeue model_overrides')
                    packEditsToJson(node);
                }
            }
        };
        modelSelect.addEventListener('change', persistModelOverride);

        const cnetpipeSelect = document.createElement("select");
        cnetpipeSelect.className = "comfy-multiline-input";
        styleTileOverrideSelect(cnetpipeSelect, "36px");
        cnetpipeSelect.placeholder = "cnetpipe " + text.textContent;
        cnetpipeSelect.dataId = "tile " + index;
        cnetpipeSelect.dataNodeId = node.id;
        cnetpipeSelect.dataset.tbgRole = "cnetpipe_override";
        const currentCNetPipeOverride = normalizeTileCNetPipeOverride(cnetpipeOverride);
        for (const optionValue of TBG_TILE_CNETPIPE_OPTIONS) {
            const option = document.createElement("option");
            option.value = optionValue;
            option.textContent = TBG_TILE_CNETPIPE_LABELS[optionValue] || optionValue;
            cnetpipeSelect.appendChild(option);
        }
        cnetpipeSelect.value = currentCNetPipeOverride || "normal";

        const persistCNetPipeOverride = () => {
            const msg = getMsg(node);
            const v = normalizeTileCNetPipeOverride(cnetpipeSelect.value);
            const uiValue = v || "normal";
            if (cnetpipeSelect.value !== uiValue) cnetpipeSelect.value = uiValue;
            if (msg.cnetpipe_overrides[index] !== v) {
                msg.cnetpipe_overrides[index] = v;
                setNodePropertySafe(node, propKeyCNetPipe(index), v);
                if (bumpRequeue(node)) {
                    console.log('[TBG] JSON bumpRequeue cnetpipe_overrides')
                    packEditsToJson(node);
                }
            }
        };
        cnetpipeSelect.addEventListener('change', persistCNetPipeOverride);

        const colorMatchSelect = document.createElement("select");
        colorMatchSelect.className = "comfy-multiline-input";
        styleTileOverrideSelect(colorMatchSelect, "40px");
        colorMatchSelect.placeholder = "color match " + text.textContent;
        colorMatchSelect.dataId = "tile " + index;
        colorMatchSelect.dataNodeId = node.id;
        colorMatchSelect.dataset.tbgRole = "color_match_override";
        const currentColorMatchOverride = normalizeTileColorMatchOverride(colorMatchOverride);
        for (const optionValue of TBG_TILE_COLOR_MATCH_OPTIONS) {
            const option = document.createElement("option");
            option.value = optionValue;
            option.textContent = TBG_TILE_COLOR_MATCH_LABELS[optionValue] || optionValue;
            colorMatchSelect.appendChild(option);
        }
        colorMatchSelect.value = colorMatchUiValue(currentColorMatchOverride);

        const persistColorMatchOverride = () => {
            const msg = getMsg(node);
            const v = normalizeTileColorMatchOverride(colorMatchSelect.value);
            const uiValue = colorMatchUiValue(v);
            if (colorMatchSelect.value !== uiValue) colorMatchSelect.value = uiValue;
            if (msg.color_match_overrides[index] !== v) {
                msg.color_match_overrides[index] = v;
                setNodePropertySafe(node, propKeyColorMatch(index), v);
                if (bumpRequeue(node)) {
                    console.log('[TBG] JSON bumpRequeue color_match_overrides')
                    packEditsToJson(node);
                }
            }
        };
        colorMatchSelect.addEventListener('change', persistColorMatchOverride);


        const textarea = document.createElement("textarea");
        textarea.style.opacity = 0.6;
        textarea.style.flexGrow = 1;
        textarea.style.height = "100%";
        textarea.className = "comfy-multiline-input";
        textarea.value = prompt || "";
        textarea.placeholder = "tile " + text.textContent;
        textarea.dataId = "tile " + index;
        textarea.dataNodeId = node.id;

        const ignoreGeneralPromptLabel = document.createElement("label");
        ignoreGeneralPromptLabel.style.display = "flex";
        ignoreGeneralPromptLabel.style.alignItems = "center";
        ignoreGeneralPromptLabel.style.gap = "6px";
        ignoreGeneralPromptLabel.style.height = "24px";
        ignoreGeneralPromptLabel.style.minHeight = "24px";
        ignoreGeneralPromptLabel.style.fontSize = "11px";
        ignoreGeneralPromptLabel.style.lineHeight = "1";
        ignoreGeneralPromptLabel.style.opacity = "0.8";
        ignoreGeneralPromptLabel.style.userSelect = "none";
        ignoreGeneralPromptLabel.title = "Use only this tile prompt for this tile; do not prepend the general prompt.";

        const ignoreGeneralPromptCheckbox = document.createElement("input");
        ignoreGeneralPromptCheckbox.type = "checkbox";
        ignoreGeneralPromptCheckbox.dataset.tbgRole = "ignore_general_prompt";
        ignoreGeneralPromptCheckbox.dataId = "tile " + index;
        ignoreGeneralPromptCheckbox.dataNodeId = node.id;
        const ignoreGeneralProperty = getNodeProperty(node, propKeyIgnoreGeneralPrompt(index), null);
        ignoreGeneralPromptCheckbox.checked = ignoreGeneralProperty !== null
            ? normalizeTileIgnoreGeneralPrompt(ignoreGeneralProperty)
            : normalizeTileIgnoreGeneralPrompt(getMsg(node).ignore_general_prompts?.[index]);
        ignoreGeneralPromptCheckbox.style.margin = "0";
        ignoreGeneralPromptCheckbox.style.flexShrink = "0";

        const ignoreGeneralPromptText = document.createElement("span");
        ignoreGeneralPromptText.textContent = "ignore general prompt";
        ignoreGeneralPromptText.style.whiteSpace = "nowrap";

        ignoreGeneralPromptLabel.appendChild(ignoreGeneralPromptCheckbox);
        ignoreGeneralPromptLabel.appendChild(ignoreGeneralPromptText);

        const persistIgnoreGeneralPrompt = () => {
            const msg = getMsg(node);
            const v = !!ignoreGeneralPromptCheckbox.checked;
            if (msg.ignore_general_prompts[index] !== v) {
                msg.ignore_general_prompts[index] = v;
                setNodePropertySafe(node, propKeyIgnoreGeneralPrompt(index), v ? "true" : "");
                if (bumpRequeue(node)) {
                    console.log('[TBG] JSON bumpRequeue ignore_general_prompts')
                    packEditsToJson(node);
                }
            }
        };
        ignoreGeneralPromptCheckbox.addEventListener('change', persistIgnoreGeneralPrompt);

         const persistPrompt = () => {
              const msg = getMsg(node);
              const v = textarea.value.trim();
              if (msg.prompts[index] !== v) {
                msg.prompts[index] = v;
                setNodePropertySafe(node, propKeyPrompt(index), v);

                // ONLY bump if changed AND accepted by requeue session
                if (bumpRequeue(node)) {
                  console.log('[TBG] JSON bumpRequeue prompts')
                  packEditsToJson(node);  // Only once per session
                }
              }
            };
        textarea.addEventListener('input', debounce(persistPrompt, 2000));

        textarea.addEventListener('change', persistPrompt); // Immediate save on field exit


        const denoiseInput = document.createElement("input");
        denoiseInput.style.opacity = 0.6;
        denoiseInput.style.height = "100%";
        denoiseInput.style.maxWidth = "1.8rem";
        denoiseInput.style.flexShrink = "0";
        denoiseInput.className = "comfy-multiline-input";
        denoiseInput.value = denoise || '';
        denoiseInput.placeholder = "denoise " + text.textContent;
        denoiseInput.dataId = "tile " + index;
        denoiseInput.dataNodeId = node.id;

        const persistDenoise = () => {
            const msg = getMsg(node);
            let v = denoiseInput.value.trim();
            if (!window.TBG.TBGWidgets.validateDenoiseValue(v)) v = '';
            else v = window.TBG.TBGWidgets.formatDenoiseValue(v);
            if (denoiseInput.value !== v) denoiseInput.value = v;
            if (msg.denoises[index] !== v) {
              msg.denoises[index] = v;
              setNodePropertySafe(node, propKeyDenoise(index), v);
              if (bumpRequeue(node)) {
                  console.log('[TBG] JSON bumpRequeue denoises')
                  packEditsToJson(node);  // Only once per session
                }
            }
          };
        denoiseInput.addEventListener('input', debounce(persistDenoise, 2000));

         denoiseInput.addEventListener('change', persistDenoise); // Immediate save on field exit

        // Dark placeholder on missing/broken thumbnail
        const fallbackSvg = `
        <svg xmlns="http://www.w3.org/2000/svg" width="140" height="140">
          <rect width="100%" height="100%" fill="rgb(53,53,53)" />
          <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
                font-size="12" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif" fill="#aaaaaa">[No Thumbnail]</text>
        </svg>
        `.trim();
        const fallbackSrc = "data:image/svg+xml;utf8," + encodeURIComponent(fallbackSvg);
        const img = document.createElement('img');
        img.src = hasTileImage(tile) ? imageDataToUrl(tile) : fallbackSrc;
        const maskData = getTileMaskData(tile);

        img.onerror = () => {
          // Prevent infinite loop if fallback also triggers onerror
          if (img.dataset.tbgFallbackApplied === "1") return;
          img.dataset.tbgFallbackApplied = "1";
          img.src = fallbackSrc;
        };

        const row_height = "176px";
        const inner_height = "176px";

        wrapper.style.display = "flex";
        wrapper.style.alignItems = "center";
        wrapper.style.gap = "12px";
        wrapper.style.width = "100%";
        wrapper.style.minWidth = "0";
        wrapper.style.minHeight = row_height;
        wrapper.style.maxHeight = row_height;
        wrapper.style.overflow = "hidden";

        img.style.height = inner_height;
        img.style.width = inner_height;
        img.style.maxHeight = inner_height;
        img.style.maxWidth = inner_height;
        img.style.objectFit = "contain";
        img.style.flexShrink = "0";

        const previewWrap = document.createElement("div");
        previewWrap.style.position = "relative";
        previewWrap.style.height = inner_height;
        previewWrap.style.width = inner_height;
        previewWrap.style.maxHeight = inner_height;
        previewWrap.style.maxWidth = inner_height;
        previewWrap.style.flexShrink = "0";
        previewWrap.style.background = "rgb(35,35,35)";
        previewWrap.style.overflow = "hidden";

        img.style.position = "absolute";
        img.style.inset = "0";
        img.style.height = "100%";
        img.style.width = "100%";

        previewWrap.appendChild(img);

        if (maskData) {
            const maskUrl = imageDataToUrl(maskData);
            const maskOverlay = document.createElement("div");
            maskOverlay.title = "Segment mask preview";
            maskOverlay.style.position = "absolute";
            maskOverlay.style.inset = "0";
            maskOverlay.style.pointerEvents = "none";
            maskOverlay.style.background = "rgba(128,128,128,0.5)";
            maskOverlay.style.maskImage = `url("${maskUrl}")`;
            maskOverlay.style.webkitMaskImage = `url("${maskUrl}")`;
            maskOverlay.style.maskSize = "contain";
            maskOverlay.style.webkitMaskSize = "contain";
            maskOverlay.style.maskRepeat = "no-repeat";
            maskOverlay.style.webkitMaskRepeat = "no-repeat";
            maskOverlay.style.maskPosition = "center";
            maskOverlay.style.webkitMaskPosition = "center";
            previewWrap.appendChild(maskOverlay);
        }

        const promptColumn = document.createElement("div");
        promptColumn.style.display = "flex";
        promptColumn.style.flexDirection = "column";
        promptColumn.style.gap = "6px";
        promptColumn.style.height = inner_height;
        promptColumn.style.minHeight = inner_height;
        promptColumn.style.maxHeight = inner_height;
        promptColumn.style.minWidth = "0";
        promptColumn.style.flexGrow = "1";

        textarea.style.height = "146px";
        textarea.style.minHeight = "146px";
        textarea.style.maxHeight = "146px";
        textarea.style.resize = "none";
        textarea.style.padding = "8px 10px";
        textarea.style.boxSizing = "border-box";
        textarea.style.overflow = "auto";
        textarea.style.margin = "0";
        textarea.style.flexGrow = "1";

        text.style.margin = "0";
        text.style.minWidth = "3ch";
        text.style.opacity = "0.8";
        text.style.fontWeight = "600";
        text.style.flexShrink = "0";

        inputEl.style.display = "block";
        inputEl.style.width = "100%";
        inputEl.style.minWidth = "0";
        inputEl.style.height = "186px";
        inputEl.style.setProperty("--comfy-widget-height", "186px");
        inputEl.style.setProperty("--comfy-widget-min-height", "186px");

        const rightControls = document.createElement("div");
        rightControls.style.display = "flex";
        rightControls.style.flexDirection = "column";
        rightControls.style.gap = "8px";
        rightControls.style.flexShrink = "0";
        rightControls.style.width = inner_height;

        denoiseInput.style.height = "40px";
        denoiseInput.style.maxWidth = "100%";
        denoiseInput.style.boxSizing = "border-box";
        denoiseInput.style.flexShrink = "0";
        colorMatchSelect.style.width = "50%";

        seedInput.style.height = "40px";
        seedInput.style.maxWidth = "100%";
        seedInput.style.boxSizing = "border-box";
        seedInput.style.flexShrink = "0";

        cnet_strengthInput.style.height = "40px";
        cnet_strengthInput.style.maxWidth = "100%";
        cnet_strengthInput.style.boxSizing = "border-box";
        cnet_strengthInput.style.flexShrink = "0";
        cfgInput.style.height = "40px";
        cfgInput.style.maxWidth = "100%";
        cfgInput.style.boxSizing = "border-box";
        cfgInput.style.flexShrink = "0";

        const denoiseRow = document.createElement("div");
        denoiseRow.style.display = "flex";
        denoiseRow.style.flexDirection = "row";
        denoiseRow.style.gap = "6px";
        denoiseInput.style.width = "50%";
        denoiseRow.appendChild(denoiseInput);
        denoiseRow.appendChild(colorMatchSelect);

        const seedRow = document.createElement("div");
        seedRow.style.display = "flex";
        seedRow.style.flexDirection = "column";
        seedRow.appendChild(seedInput);

        const cnet_strengthRow = document.createElement("div");
        cnet_strengthRow.style.display = "flex";
        cnet_strengthRow.style.flexDirection = "row";
        cnet_strengthRow.style.gap = "6px";
        cnet_strengthInput.style.width = "50%";
        cfgInput.style.width = "50%";
        cnet_strengthRow.appendChild(cnet_strengthInput);
        cnet_strengthRow.appendChild(cfgInput);

        const modelRow = document.createElement("div");
        modelRow.style.display = "flex";
        modelRow.style.flexDirection = "row";
        modelRow.style.gap = "6px";
        modelSelect.style.width = "50%";
        cnetpipeSelect.style.width = "50%";
        modelRow.appendChild(modelSelect);
        modelRow.appendChild(cnetpipeSelect);

        rightControls.appendChild(denoiseRow);
        rightControls.appendChild(seedRow);
        rightControls.appendChild(cnet_strengthRow);
        rightControls.appendChild(modelRow);

        wrapper.appendChild(text);
        wrapper.appendChild(previewWrap);
        promptColumn.appendChild(textarea);
        promptColumn.appendChild(ignoreGeneralPromptLabel);
        wrapper.appendChild(promptColumn);
        wrapper.appendChild(rightControls);

        inputEl.appendChild(wrapper);

        const eventController = attachTBGCanvasInteractionForwarding(inputEl);

        const widget = node.addDOMWidget(key, getTileRowWidgetType(), inputEl, {
            hideOnZoom: false,
            serialize: false,
            getHeight() { return 186; },
            getMinHeight() { return 186; },
            getValue() { return inputEl.value || ""; },
            setValue(v) { inputEl.value = v; },
        });
        widget.serialize = false;
        widget.element = inputEl;
        widget.inputEl = inputEl;
        const onRemove = widget.onRemove;
        widget.onRemove = function() {
            try { eventController.abort(); } catch (_) {}
            return onRemove?.apply(this, arguments);
        };

        //TBGUpscalerTileNodeWidget.setValue(node, widget.name, prompt);
        return widget;
    },
};

/*──────────────────────────────────────────────────────────────────────────*/
/* Reset / Clean handler                                                     */
/*──────────────────────────────────────────────────────────────────────────*/

class TBG_NodePrompter {
    static async clean(node) {
        window.TBG.TBGETUR.clean = false;
        const cleanedLabel = "";
        const nodeTitle = node.title;
        node.title = nodeTitle + cleanedLabel;

        const msg = getMsg(node);
        const inp = getInp(node);

        inp.tiles = [];
        msg.tiles = [];
        msg.prompts = [];
        inp.prompts = [];
        inp.denoises = [];
        msg.denoises = [];
        inp.seeds = [];
        msg.seeds = [];
        inp.cnet_strength = [];
        msg.cnet_strength = [];
        inp.cfg_overrides = [];
        msg.cfg_overrides = [];
        inp.model_overrides = [];
        msg.model_overrides = [];
        inp.cnetpipe_overrides = [];
        msg.cnetpipe_overrides = [];
        inp.color_match_overrides = [];
        msg.color_match_overrides = [];
        inp.ignore_general_prompts = [];
        msg.ignore_general_prompts = [];

        const MAX_CLEAR = 512;
        for (let i = 0; i < MAX_CLEAR; i++) {
            try {
                delete node.properties[propKeyPrompt(i)];
                delete node.properties[propKeyDenoise(i)];
                delete node.properties[propKeySeed(i)];
                delete node.properties[propKeyCNet(i)];
                delete node.properties[propKeyCFG(i)];
                delete node.properties[propKeyModel(i)];
                delete node.properties[propKeyCNetPipe(i)];
                delete node.properties[propKeyColorMatch(i)];
                delete node.properties[propKeyIgnoreGeneralPrompt(i)];
            } catch (_) {}
        }

        setNodePropertySafe(node, "tile_edits_json", "");
        console.log('[TBG] async clean(node)')
        packEditsToJson(node, true);

        TBGUpscalerTileNodeWidget.setValue(node, TBGUpscalerTileNodeWidget.INDEX.name, TBGUpscalerTileNodeWidget.INDEX.default);
        TBGUpscalerTileNodeWidget.refresh(node);

        setTimeout(() => { node.title = nodeTitle; }, 500);
    }
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Widget Manager                                                            */
/*──────────────────────────────────────────────────────────────────────────*/

class TBGUpscalerTileNodeWidget {
    static INDEX = { name: "Filter by Indexes", default: "" };
    static PROMPT = { name: "Prompt", default: "" };
    static DENOISE = { name: "Denoise", values: ['unchanged', 'Use Global Denoise'], default: "unchanged", min: 0.00, max: 1.00, step: 0.01 };
    static SEED = { name: "Seed", default: "" };
    static CNET_STRENGTH = { name: "CNet Strength", values: ['unchanged', 'Use Global CNet Strength'], default: "unchanged", min: 0.00, max: 1.00, step: 0.01 };
    static CLEAN = { name: 'Reset', default: false };

    static getByName(node, name) {
        return node.widgets?.find((w) => w.name === name);
    }
    static setValue(node, name, value) {
        const nodeWidget = this.getByName(node, name);
        if (nodeWidget) {
            nodeWidget.value = value;
            node.setProperty(name, nodeWidget?.value ?? node.properties[name]);
            node.setDirtyCanvas(true);
        }
    }


    static refresh(node) {
        const msg = getMsg(node);

        // Remove legacy "tile N" text widgets and any previous TBG tile row wrappers.
        node.widgets = node.widgets.filter(widget => {
            if (widget.type === "text" && /^tile\s+\d+$/i.test(widget.name)) {
                widget.onRemove?.();
                return false;
            }
            if (isTBGTileRowWidget(widget) || isLegacyTileRowWidget(widget)) {
                try { widget.onRemove?.(); } catch (_) {}
                return false;
            }
            return true;
        });

        // Header controls
        this.setIndexInput(node);
        this.setPromptInput(node);
        this.setDenoiseInput(node);
        this.setSeedInput(node);
        this.setcnet_strengthInput(node);
        requestNodeRedraw(node);

        // Tile rows
        this.setPrompterInputs(node);

        // Reset toggle only if rows exist
        if (msg.prompts.length) this.setCleanSwitch(node);

        fitTileOverrideNodeHeight(node, { allowShrink: true });

        // Hide stray legacy "tile N" widgets
        try {
            for (const w of node.widgets) {
                if (w.type === "text" && /^tile\s+\d+$/i.test(w.name || "")) {
                    w.hidden = true;
                    if (w.options) w.options.disabled = true;
                }
            }
            fitTileOverrideNodeHeight(node, { allowShrink: true });
        } catch (_) {}
    }

    static init(node) {
        this.setIndexInput(node);
        this.setPromptInput(node);
        this.setDenoiseInput(node);
        this.setSeedInput(node);
        this.setcnet_strengthInput(node);
        this.setCleanSwitch(node);
    }

    static setIndexInput(node) {
        const nodeWidget = this.getByName(node, this.INDEX.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "text",
                this.INDEX.name,
                this.INDEX.default,
                (value) => {
                    this.setValue(node, this.INDEX.name, value);
                    this.setValue(node, this.PROMPT.name, this.PROMPT.default);
                    this.setValue(node, this.DENOISE.name, this.DENOISE.default);
                    this.refresh(node);
                }, {}
            );
            this.setValue(node, this.INDEX.name, this.INDEX.default);
        }
    }

    static setPromptInput(node) {
        const nodeWidget = this.getByName(node, this.PROMPT.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "text",
                this.PROMPT.name,
                this.PROMPT.default,
                (value) => {
                    const msg = getMsg(node);
                    this.setValue(node, this.PROMPT.name, value);
                    const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;

                    node.widgets = node.widgets.filter((widget) => {
                        if (isTBGTileRowWidget(widget)) {
                            const textarea = getTileRowWidgetElement(widget).querySelector('textarea[placeholder^="tile "]');
                            if (textarea) {
                                const dataId = textarea.getAttribute('placeholder');
                                const indexValue = parseInt(dataId.replace('tile ', ''), 10);
                                const realIndexValue = indexValue - 1;
                                const index_filtered = input_list.split(",").map(num => Number(num) - 1);
                                const inFilter = (input_list !== "" && index_filtered.indexOf(realIndexValue) > -1) || input_list === "";
                                if (inFilter) {
                                    const v = (value || "").trim();
                                    textarea.value = v;
                                    setNodePropertySafe(node, propKeyPrompt(realIndexValue), v);
                                    msg.prompts[realIndexValue] = v;
                                   if (bumpRequeue(node)) {
                                        console.log('[TBG] bumpRequeue textarea')
                                      packEditsToJson(node);  // Only once per session
                                    }
                                }
                            }
                        }
                        return true;
                    });

                    this.setValue(node, this.PROMPT.name, this.PROMPT.default);
                }, {}
            );
            this.setValue(node, this.PROMPT.name, this.PROMPT.default);
        }
    }

    static setDenoiseInput(node) {
        const nodeWidget = this.getByName(node, this.DENOISE.name);
        if (nodeWidget == undefined) {
            window.TBG.TBGETUR.params.denoise.values.length = 0;
            for (let i = this.DENOISE.min; i <= this.DENOISE.max; i = parseFloat((i + this.DENOISE.step).toFixed(2))) {
                window.TBG.TBGETUR.params.denoise.values.push(i.toFixed(2));
            }

            node.addWidget(
                "combo",
                this.DENOISE.name,
                this.DENOISE.default,
                (value) => {
                    const msg = getMsg(node);
                    if (value !== "unchanged") {
                        if (value === 'Use Global Denoise') value = '';
                        const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;
                        node.widgets = node.widgets.filter((widget) => {
                            if (isTBGTileRowWidget(widget)) {
                                const input = getTileRowWidgetElement(widget).querySelector('input[placeholder^="denoise "]');
                                if (input) {
                                    const dataId = input.getAttribute('placeholder');
                                    const indexValue = parseInt(dataId.replace('denoise ', ''), 10);
                                    const realIndexValue = indexValue - 1;
                                    const index_filtered = input_list.split(",").map(num => Number(num) - 1);
                                    const inFilter = (input_list !== "" && index_filtered.indexOf(realIndexValue) > -1) || input_list === "";
                                    if (inFilter) {
                                        let v = String(value).trim();
                                        if (!window.TBG.TBGWidgets.validateDenoiseValue(v)) v = '';
                                        else v = window.TBG.TBGWidgets.formatDenoiseValue(v);
                                        input.value = v;
                                        setNodePropertySafe(node, propKeyDenoise(realIndexValue), v);
                                        msg.denoises[realIndexValue] = v;
                                        console.log('[TBG] addWidget setDenoiseInput')
                                        packEditsToJson(node);
                                    }
                                }
                            }
                            return true;
                        });
                    }
                }, {
                    values: window.TBG.TBGETUR.params.denoise.values
                }
            );
            this.setValue(node, this.DENOISE.name, this.DENOISE.default);
        }
    }

    static setcnet_strengthInput(node) {
        const nodeWidget = this.getByName(node, this.CNET_STRENGTH.name);
        if (nodeWidget == undefined) {
            window.TBG.TBGETUR.params.cnet_strength.values.length = 0;
            for (let i = this.CNET_STRENGTH.min; i <= this.CNET_STRENGTH.max; i = parseFloat((i + this.CNET_STRENGTH.step).toFixed(2))) {
                window.TBG.TBGETUR.params.cnet_strength.values.push(i.toFixed(2));
            }

            node.addWidget(
                "combo",
                this.CNET_STRENGTH.name,
                this.CNET_STRENGTH.default,
                (value) => {
                    const msg = getMsg(node);
                    if (value !== "unchanged") {
                        if (value === 'Use Global CNet Strength') value = '';
                        const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;

                        node.widgets = node.widgets.filter((widget) => {
                            if (isTBGTileRowWidget(widget)) {
                                const input = getTileRowWidgetElement(widget).querySelector('input[placeholder^="cnet_strength "]');
                                if (input) {
                                    const dataId = input.getAttribute('placeholder');
                                    const indexValue = parseInt(dataId.replace('cnet_strength ', ''), 10);
                                    const realIndexValue = indexValue - 1;
                                    const index_filtered = input_list.split(",").map(num => Number(num) - 1);
                                    const inFilter = (input_list !== "" && index_filtered.indexOf(realIndexValue) > -1) || input_list === "";
                                    if (inFilter) {
                                        let v = String(value).trim();
                                        if (!window.TBG.TBGWidgets.validateDenoiseValue(v)) v = '';
                                        else v = window.TBG.TBGWidgets.formatDenoiseValue(v);
                                        input.value = v;
                                        setNodePropertySafe(node, propKeyCNet(realIndexValue), v);
                                        msg.cnet_strength[realIndexValue] = v;
                                        console.log('[TBG] addWidget cnet_strength')
                                        packEditsToJson(node);
                                    }
                                }
                            }
                            return true;
                        });
                    }
                }, {
                    values: window.TBG.TBGETUR.params.cnet_strength.values
                }
            );
            this.setValue(node, this.CNET_STRENGTH.name, this.CNET_STRENGTH.default);
        }
    }

    static setSeedInput(node) {
        const nodeWidget = this.getByName(node, this.SEED.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "text",
                this.SEED.name,
                this.SEED.default,
                (value) => {
                    const msg = getMsg(node);
                    this.setValue(node, this.SEED.name, value);
                    const input_list = node.properties[this.INDEX.name] ?? this.INDEX.default;
                    const index_filtered = input_list
                        .split(",")
                        .filter((s) => s.trim() !== "")
                        .map((num) => Number(num) - 1);

                    node.widgets = node.widgets.filter((widget) => {
                        if (isTBGTileRowWidget(widget)) {
                            const seedInput = getTileRowWidgetElement(widget).querySelector('input[placeholder^="seed "]');
                            if (seedInput) {
                                const dataId = seedInput.getAttribute('placeholder');
                                const indexValue = parseInt(dataId.replace('seed ', ''), 10);
                                const realIndexValue = indexValue - 1;
                                const inFilter = (input_list !== "" && index_filtered.indexOf(realIndexValue) > -1) || input_list === "";
                                if (inFilter) {
                                    seedInput.value = value;
                                    let v = String(value).trim();
                                    if (!window.TBG.TBGWidgets.validateSeedValue(v)) v = '';
                                    else v = window.TBG.TBGWidgets.formatSeedValue(v);
                                    setNodePropertySafe(node, propKeySeed(realIndexValue), v);
                                    msg.seeds[realIndexValue] = v;
                                    console.log('[TBG] addWidget seeds')
                                    packEditsToJson(node);
                                }
                            }
                        }
                        return true;
                    });

                    this.setValue(node, this.SEED.name, this.SEED.default);
                }, {}
            );
            this.setValue(node, this.SEED.name, this.SEED.default);
        }
    }

    static setCleanSwitch(node) {
        const nodeWidget = this.getByName(node, this.CLEAN.name);
        if (nodeWidget == undefined) {
            node.addWidget(
                "toggle",
                this.CLEAN.name,
                this.CLEAN.default,
                () => {
                    this.setValue(node, this.CLEAN.name, this.CLEAN.default);
                    TBG_NodePrompter.clean(node);
                }, {}
            );
            this.setValue(node, this.CLEAN.name, this.CLEAN.default);
        }
    }

    static setPrompterInputs(node) {
        const msg = getMsg(node);

        syncNodeWidgetsToTileState(node);

        node.widgets = node.widgets.filter(widget => {
            if (isTBGTileRowWidget(widget) || isLegacyTileRowWidget(widget)) {
                try { widget.onRemove?.(); } catch (_) {}
                return false;
            }
            return true;
        });

        const count = getTileRowCount(msg);
        msg.prompts.length = count;
        msg.denoises.length = count;
        msg.seeds.length = count;
        msg.cnet_strength.length = count;
        msg.cfg_overrides.length = count;
        msg.model_overrides.length = count;
        msg.cnetpipe_overrides.length = count;
        msg.color_match_overrides.length = count;
        msg.ignore_general_prompts.length = count;

        for (let i = 0; i < count; i++) {
            const promptVal = getNodeProperty(node, `prompt_${i}`, "");
            const denoiseVal = getNodeProperty(node, `denoise_${i}`, "");
            const seedVal = getNodeProperty(node, `seed_${i}`, "");
            const cnetVal = getNodeProperty(node, `cnet_strength_${i}`, "");
            const cfgVal = getNodeProperty(node, `cfg_override_${i}`, "");
            const modelVal = normalizeTileModelOverride(getNodeProperty(node, `model_override_${i}`, ""));
            const cnetpipeVal = normalizeTileCNetPipeOverride(getNodeProperty(node, `cnetpipe_override_${i}`, ""));
            const colorMatchVal = normalizeTileColorMatchOverride(getNodeProperty(node, `color_match_override_${i}`, ""));
            const ignoreGeneralVal = normalizeTileIgnoreGeneralPrompt(getNodeProperty(node, `ignore_general_prompt_${i}`, false));

            msg.prompts[i] = promptVal;
            msg.denoises[i] = denoiseVal;
            msg.seeds[i] = seedVal;
            msg.cnet_strength[i] = cnetVal;
            msg.cfg_overrides[i] = cfgVal;
            msg.model_overrides[i] = modelVal;
            msg.cnetpipe_overrides[i] = cnetpipeVal;
            msg.color_match_overrides[i] = colorMatchVal;
            msg.ignore_general_prompts[i] = ignoreGeneralVal;

            window.TBG.TBGWidgets.WRAPPER(
                `tile_${i}`,
                i,
                promptVal,
                msg.tiles[i] || null,
                denoiseVal,
                seedVal,
                cnetVal,
                cfgVal,
                modelVal,
                cnetpipeVal,
                colorMatchVal,
                node
            );
        }
            console.log('[TBG] JSON setPrompterInputs window.TBG.TBGWidgets.WRAPPER');

            // Only mirror if we actually have something meaningful
            const hasContent = [...(msg.prompts || []),
                                 ...(msg.denoises || []),
                                 ...(msg.seeds || []),
                                 ...(msg.cnet_strength || []),
                                 ...(msg.cfg_overrides || []),
                                 ...(msg.model_overrides || []),
                                 ...(msg.cnetpipe_overrides || []),
                                 ...(msg.color_match_overrides || [])]
                .some(v => v != null && String(v).trim() !== "") ||
                (msg.ignore_general_prompts || []).some(Boolean);

            if (hasContent) {
                packEditsToJson(node);
            }
    }
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Status overlay + extension registration                                  */
/*──────────────────────────────────────────────────────────────────────────*/

const myExtension = {
    name: "ComfyUI.TBG.TilePrompter",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const nodeName = String(nodeData?.name ?? "");
        const nodeDisplayName = String(nodeData?.display_name ?? "");
        const nodeComfyClass = String(nodeType?.comfyClass ?? "");
        const isTileOverridesNode = isTileOverridesNodeClass(nodeType, nodeData) ||
            nodeName === "TBG ETUR Tile Overrides" ||
            nodeDisplayName === "TBG ETUR Tile Overrides" ||
            nodeComfyClass === "TBG ETUR Tile Overrides" ||
            nodeComfyClass === "TBG_TilePrompter_v1";

        if (isTileOverridesNode) {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {

              const r = onExecuted?.apply(this, arguments);

                const prompts = [];
                const denoises = [];
                const seeds = [];
                const cnet = [];
                const cfgOverrides = [];
                const modelOverrides = [];
                const cnetpipeOverrides = [];
                const colorMatchOverrides = [];
                const ignoreGeneralPrompts = [];
                const msg = getMsg(this);
                const tiles = getTileListFromExecutionMessage(message);
                const tilerContextKey = getTilerContextKeyFromExecutionMessage(message);
                const tileCount = tiles.length;
                msg.tiles = tiles;
                if (tilerContextKey) {
                    msg.tiler_context_key = String(tilerContextKey);
                }
                if (tileCount > 0) {
                    trimTileStateToCount(this, tileCount);
                }

                const wrappers = getTileRowWidgetElements(this);

                wrappers.forEach(wrapperEl => {
                    const textarea = wrapperEl.querySelector('textarea[placeholder^="tile "]');
                    if (!textarea) return;
                    const idx = parseInt(textarea.placeholder.replace(/\D/g, ""), 10) - 1;
                    if (tileCount > 0 && idx >= tileCount) return;

                    const denInput = wrapperEl.querySelector('input[placeholder^="denoise "]');
                    const seedInput = wrapperEl.querySelector('input[placeholder^="seed "]');
                    const cnetInput = wrapperEl.querySelector('input[placeholder^="cnet_strength "]');
                    const cfgInput = wrapperEl.querySelector('input[placeholder^="cfg "]');
                    const modelSelect = wrapperEl.querySelector('select[data-tbg-role="model_override"]');
                    const cnetpipeSelect = wrapperEl.querySelector('select[data-tbg-role="cnetpipe_override"]');
                    const colorMatchSelect = wrapperEl.querySelector('select[data-tbg-role="color_match_override"]');
                    const ignoreGeneralCheckbox = wrapperEl.querySelector('input[data-tbg-role="ignore_general_prompt"]');

                    prompts[idx] = textarea.value || "";
                    denoises[idx] = denInput?.value || "";
                    seeds[idx] = seedInput?.value || "";
                    cnet[idx] = cnetInput?.value || "";
                    cfgOverrides[idx] = cfgInput?.value || "";
                    modelOverrides[idx] = normalizeTileModelOverride(modelSelect?.value || "");
                    cnetpipeOverrides[idx] = normalizeTileCNetPipeOverride(cnetpipeSelect?.value || "");
                    colorMatchOverrides[idx] = normalizeTileColorMatchOverride(colorMatchSelect?.value || "");
                    ignoreGeneralPrompts[idx] = !!ignoreGeneralCheckbox?.checked;

                    setNodePropertySafe(this, `prompt_${idx}`, (prompts[idx] || "").trim());
                    setNodePropertySafe(this, `denoise_${idx}`, (denoises[idx] || "").trim());
                    setNodePropertySafe(this, `seed_${idx}`, (seeds[idx] || "").trim());
                    setNodePropertySafe(this, `cnet_strength_${idx}`, (cnet[idx] || "").trim());
                    setNodePropertySafe(this, `cfg_override_${idx}`, (cfgOverrides[idx] || "").trim());
                    setNodePropertySafe(this, `model_override_${idx}`, (modelOverrides[idx] || "").trim());
                    setNodePropertySafe(this, `cnetpipe_override_${idx}`, (cnetpipeOverrides[idx] || "").trim());
                    setNodePropertySafe(this, `color_match_override_${idx}`, (colorMatchOverrides[idx] || "").trim());
                    setNodePropertySafe(this, `ignore_general_prompt_${idx}`, ignoreGeneralPrompts[idx] ? "true" : "");
                });

                (async () => {
                    try {
                        const res = await fetch(`/TBG/get_input_prompts?node=${this.id}`);
                        if (!res.ok) return;
                        const {prompts_in = []} = await res.json();

                        prompts_in.slice(0, tileCount || prompts_in.length).forEach((p, i) => {
                            if (p != "") {
                                setNodePropertySafe(this, `prompt_${i}`, p);
                                msg.prompts[i] = p;
                            }
                        });
                        packEditsToJson(this);
                        TBGUpscalerTileNodeWidget.refresh(this);
                        scheduleTileOverrideNodeHeightFit(this, { allowShrink: true });
                    } catch (err) {
                        console.error("prompt fetch failed:", err);
                    }
                })();

                msg.prompts = prompts;
                msg.denoises = denoises;
                msg.seeds = seeds;
                msg.cnet_strength = cnet;
                msg.cfg_overrides = cfgOverrides;
                msg.model_overrides = modelOverrides;
                msg.cnetpipe_overrides = cnetpipeOverrides;
                msg.color_match_overrides = colorMatchOverrides;
                msg.ignore_general_prompts = ignoreGeneralPrompts;
                if (tileCount > 0) {
                    msg.prompts.length = tileCount;
                    msg.denoises.length = tileCount;
                    msg.seeds.length = tileCount;
                    msg.cnet_strength.length = tileCount;
                    msg.cfg_overrides.length = tileCount;
                    msg.model_overrides.length = tileCount;
                    msg.cnetpipe_overrides.length = tileCount;
                    msg.color_match_overrides.length = tileCount;
                    msg.ignore_general_prompts.length = tileCount;
                }





              // const r = onExecuted?.apply(this, arguments);

              // ONLY update message cache + refresh UI - NO node.properties mutation!
              //  const msg = getMsg(this);
              // msg.tiles = message.tiles || [];  // Cache tiles for UI

              // Refresh UI widgets with new tiles (no property writes)
              TBGUpscalerTileNodeWidget.refresh(this);
              scheduleTileOverrideNodeHeightFit(this, { allowShrink: true });
              console.warn("[TBG] Tile Promtper onExecuted sync done for node");
            // CRITICAL: Persist data now that we have tiles and UI is populated
                const hasContent = [...msg.prompts, ...msg.denoises, ...msg.seeds, ...msg.cnet_strength, ...msg.cfg_overrides, ...msg.model_overrides, ...msg.cnetpipe_overrides, ...msg.color_match_overrides]
                  .some(v => v !== null && v !== "") || (msg.ignore_general_prompts || []).some(Boolean);


              packEditsToJson(this, true);
                //if (hasContent) {
               //     console.log('[TBG] JSON onExecuted hasContent',hasContent)
                //    packEditsToJson(this, false); // Mirror to backend without forcing property write
               //    console.warn("[TBG] Tile Promtper onExecuted sync JSON");
                //} else {
                //     console.warn("[TBG] Tile Promtper onExecuted sync JSON faild hasContent=False");}
                return r;
              };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(info) {
                const r = onConfigure?.apply(this, arguments);
                setTimeout(() => refreshTileOverrideNode(this), 0);
                return r;
            };

            if (!window.TBG._loadGraphHooked) {
              const origLoadGraphData = app.loadGraphData;
              app.loadGraphData = async function(data, ...args) {
                const r = await origLoadGraphData.apply(this, [data, ...args]);
                setTimeout(() => refreshAllTileOverrideNodes({ pack: true }), 0);
                return r;
              };
              window.TBG._loadGraphHooked = true;
            }


            if (!window.TBG._onNodeCreated) {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function() {

                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                    this.widgets = getNodeWidgets(this).filter((widget) => {
                        if (widget.type === "text" && widget.name?.startsWith?.("tile ")) {
                            widget.onRemove?.();
                            return false;
                        }
                        if (isTBGTileRowWidget(widget) || isLegacyTileRowWidget(widget)) {
                            try { widget.onRemove?.(); } catch (_) {}
                            return false;
                        }
                        return true;
                    });

                    try {
                        const msg = getMsg(this);
                        const prompts = [];
                        const denoises = [];
                        const seeds = [];
                        const cnet = [];
                        const cfgOverrides = [];
                        const modelOverrides = [];
                        const cnetpipeOverrides = [];
                        const colorMatchOverrides = [];
                        const ignoreGeneralPrompts = [];
                        const tileCountFromImages = Array.isArray(msg.tiles) ? msg.tiles.length : 0;
                        let tileCountFromProperties = 0;
                        for (let i = 0; i < 512; i++) {
                            const hasAnyProperty = [
                                getNodeProperty(this, propKeyPrompt(i), null),
                                getNodeProperty(this, propKeyDenoise(i), null),
                                getNodeProperty(this, propKeySeed(i), null),
                                getNodeProperty(this, propKeyCNet(i), null),
                                getNodeProperty(this, propKeyCFG(i), null),
                                getNodeProperty(this, propKeyModel(i), null),
                                getNodeProperty(this, propKeyCNetPipe(i), null),
                                getNodeProperty(this, propKeyColorMatch(i), null),
                                getNodeProperty(this, propKeyIgnoreGeneralPrompt(i), null)
                            ].some(prop => prop !== null);
                            if (hasAnyProperty) tileCountFromProperties = i + 1;
                        }
                        const actualTileCount = tileCountFromImages > 0 ? tileCountFromImages : tileCountFromProperties;

                        for (let i = 0; i < actualTileCount; i++) {
                            prompts[i] = getNodeProperty(this, propKeyPrompt(i), '');
                            denoises[i] = getNodeProperty(this, propKeyDenoise(i), '');
                            seeds[i] = getNodeProperty(this, propKeySeed(i), '');
                            cnet[i] = getNodeProperty(this, propKeyCNet(i), '');
                            cfgOverrides[i] = getNodeProperty(this, propKeyCFG(i), '');
                            modelOverrides[i] = normalizeTileModelOverride(getNodeProperty(this, propKeyModel(i), ''));
                            cnetpipeOverrides[i] = normalizeTileCNetPipeOverride(getNodeProperty(this, propKeyCNetPipe(i), ''));
                            colorMatchOverrides[i] = normalizeTileColorMatchOverride(getNodeProperty(this, propKeyColorMatch(i), ''));
                            ignoreGeneralPrompts[i] = normalizeTileIgnoreGeneralPrompt(getNodeProperty(this, propKeyIgnoreGeneralPrompt(i), false));
                        }

                        if (tileCountFromImages > 0) {
                            clearExcessTileProperties(this, actualTileCount);
                        }

                        const mergePrefProp = (propArr, cacheArr = []) => {
                            const n = Math.max(propArr.length, cacheArr.length);
                            const out = new Array(n);
                            for (let k = 0; k < n; k++) {
                                out[k] = (propArr[k] !== undefined && propArr[k] !== null) ? propArr[k] : (cacheArr[k] ?? '');
                            }
                            return out;
                        };

                        msg.prompts = mergePrefProp(prompts, msg.prompts);
                        msg.denoises = mergePrefProp(denoises, msg.denoises);
                        msg.seeds = mergePrefProp(seeds, msg.seeds);
                        msg.cnet_strength = mergePrefProp(cnet, msg.cnet_strength);
                        msg.cfg_overrides = mergePrefProp(cfgOverrides, msg.cfg_overrides);
                        msg.model_overrides = mergePrefProp(modelOverrides, msg.model_overrides).map(normalizeTileModelOverride);
                        msg.cnetpipe_overrides = mergePrefProp(cnetpipeOverrides, msg.cnetpipe_overrides).map(normalizeTileCNetPipeOverride);
                        msg.color_match_overrides = mergePrefProp(colorMatchOverrides, msg.color_match_overrides).map(normalizeTileColorMatchOverride);
                        msg.ignore_general_prompts = mergePrefProp(ignoreGeneralPrompts, msg.ignore_general_prompts).map(normalizeTileIgnoreGeneralPrompt);

                        const inp = getInp(this);
                        inp.prompts = msg.prompts.slice();
                        inp.denoises = msg.denoises.slice();
                        inp.seeds = msg.seeds.slice();
                        inp.cnet_strength = msg.cnet_strength.slice();
                        inp.cfg_overrides = msg.cfg_overrides.slice();
                        inp.model_overrides = msg.model_overrides.slice();
                        inp.cnetpipe_overrides = msg.cnetpipe_overrides.slice();
                        inp.color_match_overrides = msg.color_match_overrides.slice();
                        inp.ignore_general_prompts = msg.ignore_general_prompts.slice();
                    } catch (_) {}

                    TBGUpscalerTileNodeWidget.init(this);
                    this.onResize?.(this.size);
                    console.log('[TBG] JSON onNodeCreated this:',this)
                    setTimeout(() => packEditsToJson(this), 10);  // Just final JSON sync

                    return r;
                };
                window.TBG._onNodeCreated = true;
            }
        } else {
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                const r = getExtraMenuOptions?.apply?.(this, arguments);

                let img;
                if (this.imageIndex != null) img = this.imgs[this.imageIndex];
                else if (this.overIndex != null) img = this.imgs[this.overIndex];

                if (img) {
                    let pos = options.findIndex((o) => o.content === "Save Image");
                    if (pos === -1) pos = 0;
                    else pos++;
                    options.splice(pos, 0, {
                        content: "TilePrompt (TBG)",
                        callback: async () => {
                            let src = img.src;
                            src = src.replace(
                                "/view?",
                                `/TBG/tile_prompt?node=${this.id}&clientId=${api.clientId}&`
                            );
                            const res = await (await fetch(src)).json();
                            alert(res);
                        },
                    });
                }

                return r;
            };
    }},

    async nodeCreated(node) {
        if (!isTileOverrideRuntimeNode(node)) return;
        setTimeout(() => refreshTileOverrideNode(node), 0);
    },

    async loadedGraphNode(node) {
        if (!isTileOverrideRuntimeNode(node)) return;
        setTimeout(() => refreshTileOverrideNode(node, { pack: true }), 0);
    },

    async afterConfigureGraph() {
        setTimeout(() => refreshAllTileOverrideNodes(), 0);
    },
};



app.registerExtension(myExtension);

let lastTileRowMode = getTileRowMode();
setInterval(() => {
    const nextMode = getTileRowMode();
    if (nextMode === lastTileRowMode) return;
    lastTileRowMode = nextMode;
    if (!app?.graph?._nodes) return;
    for (const node of app.graph._nodes) {
        if (!isTileOverrideRuntimeNode(node)) continue;
        syncNodeWidgetsToTileState(node);
        refreshTileOverrideNode(node, { pack: true });
    }
}, 1000);
