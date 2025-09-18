/*******************************************************************************************
 *
 * McBoaty Tile-Prompter – MULTI-INSTANCE version for TBG enhanced upscaler and refiner pro
 *
 * PURPOSE
 * ───────
 * Provide a "Tile Prompter" node that shows one row per image tile with:
 *  – prompt textarea
 *  – denoise value
 *  – seed
 *  – ControlNet strength
 *  – live tile preview
 * Persist user edits to node.properties and a packed JSON string
 * Offer header-level bulk inputs (apply same prompt/denoise/seed/CNet to many rows)
 * Send status tags back from backend via custom event
 *
 * CHANGES vs your original
 * ────────────────────────
 * • All runtime caches are now PER-NODE, keyed by node.id:
 *     getMsg(node) / getInp(node)
 * • No shared global message/inputs arrays; multiple TilePrompters co-exist independently.
 * • All references to window.TBG.TBGETUR.message / .inputs swapped to the per-node caches.
 *
 * Copyright (c) 2025 Tobias Laarmann
 * Copyright (c) 2024 David Asquiedge
 * Derivative of "McBoaty_v5.js" (c) 2024 David Asquiedge – MIT License with attribution.
 *
 ******************************************************************************************/

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";   // eslint-disable-line no-unused-vars
import { api } from "../../scripts/api.js";

/*──────────────────────────────────────────────────────────────────────────*/
/* Backend connection status (unchanged; informational)                     */
/*──────────────────────────────────────────────────────────────────────────*/

api.addEventListener("status", (event) => {
    console.log("[TBG_upscaler2] status event:", event.detail);
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
  console.warn('[TBG] backend went down – waiting…');
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
        messages: {},   // node.id → {prompts, tiles, denoises, seeds, cnet_strength}
        inputs:   {},   // node.id → {prompts, tiles, denoises, seeds, cnet_strength}

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
    if (!ns.messages[id]) ns.messages[id] = { prompts: [], tiles: [], denoises: [], seeds: [], cnet_strength: [] };
    if (!ns.inputs[id])   ns.inputs[id]   = { prompts: [], tiles: [], denoises: [], seeds: [], cnet_strength: [] };
    return id;
}

function getMsg(node) {
    const id = ensureNodeStores(node);
    return id ? window.TBG.TBGETUR.messages[id] : { prompts: [], tiles: [], denoises: [], seeds: [], cnet_strength: [] };
}

function getInp(node) {
    const id = ensureNodeStores(node);
    return id ? window.TBG.TBGETUR.inputs[id] : { prompts: [], tiles: [], denoises: [], seeds: [], cnet_strength: [] };
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Property key builders + safe property access                             */
/*──────────────────────────────────────────────────────────────────────────*/

function propKeyPrompt(i) {return `prompt_${i}`;}
function propKeyDenoise(i) {return `denoise_${i}`;}
function propKeySeed(i) {return `seed_${i}`;}
function propKeyCNet(i) {return `cnet_strength_${i}`;}

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

function imageDataToUrl(data) {
  return api.apiURL(
    `/view?filename=${encodeURIComponent(data.filename)}&type=${data.type}&subfolder=${data.subfolder}${app.getPreviewFormatParam()}${app.getRandParam()}`
  );
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

function clearExcessTileProperties(node, currentTileCount) {
    const MAX_CLEAR = 512;
    for (let i = currentTileCount; i < MAX_CLEAR; i++) {
        const hasAnyProperty = [
            getNodeProperty(node, `prompt_${i}`, null),
            getNodeProperty(node, `denoise_${i}`, null), 
            getNodeProperty(node, `seed_${i}`, null),
            getNodeProperty(node, `cnet_strength_${i}`, null)
        ].some(prop => prop !== null);
        if (!hasAnyProperty) break;
        try {
            delete node.properties[propKeyPrompt(i)];
            delete node.properties[propKeyDenoise(i)];
            delete node.properties[propKeySeed(i)];
            delete node.properties[propKeyCNet(i)];
        } catch (_) {}
    }
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Per-node: properties → message caches                                     */
/*──────────────────────────────────────────────────────────────────────────*/

function syncPropertiesToMessage(node) {
    const msg = getMsg(node);
    const prompts = [];
    const denoises = [];
    const seeds = [];
    const cnet = [];
    
    const MAX = 512;
    let actualTileCount = 0;

    for (let i = 0; i < MAX; i++) {
        const p = getNodeProperty(node, `prompt_${i}`, null);
        const d = getNodeProperty(node, `denoise_${i}`, null);
        const s = getNodeProperty(node, `seed_${i}`, null);
        const c = getNodeProperty(node, `cnet_strength_${i}`, null);
        
        if (p !== null || d !== null || s !== null || c !== null) {
            actualTileCount = i + 1;
            prompts[i] = p ?? "";
            denoises[i] = d ?? "";
            seeds[i] = s ?? "";
            cnet[i] = c ?? "";
        } else if (i > 0 && actualTileCount === 0) {
            break;
        }
    }

    clearExcessTileProperties(node, actualTileCount);

    msg.prompts = prompts;
    msg.denoises = denoises;
    msg.seeds = seeds;
    msg.cnet_strength = cnet;
    
    if (actualTileCount > 0 && (!Array.isArray(msg.tiles) || msg.tiles.length < actualTileCount)) {
        const existingTiles = Array.isArray(msg.tiles) ? msg.tiles : [];
        msg.tiles = Array.from({ length: actualTileCount }, (_, i) => {
            return existingTiles[i] || { filename: "", type: "", subfolder: "" };
        });
    }
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Persist wrappers → properties → per-node caches                           */
/*──────────────────────────────────────────────────────────────────────────*/

function syncAllWrappersToProperties(node) {
    const wrappers = (node.widgets || [])
        .filter(w => w.type === "customtext" && w.inputEl)
        .map(w => w.inputEl);

    const msg = getMsg(node);

    wrappers.forEach(wrapperEl => {
        const textarea = wrapperEl.querySelector('textarea[placeholder^="tile "]');
        if (!textarea) return;

        const idx = parseInt(textarea.placeholder.replace(/\D/g, ""), 10) - 1;

        const denoiseInput = wrapperEl.querySelector('input[placeholder^="denoise "]');
        const seedInput = wrapperEl.querySelector('input[placeholder^="seed "]');
        const cnetInput = wrapperEl.querySelector('input[placeholder^="cnet_strength "]');

        const rawD = (denoiseInput?.value || "").trim();
        const rawS = (seedInput?.value || "").trim();
        const rawC = (cnetInput?.value || "").trim();

        const normD = rawD === '' ? '' :
            (window.TBG.TBGWidgets.validateDenoiseValue(rawD) ? window.TBG.TBGWidgets.formatDenoiseValue(rawD) : '');
        const normS = rawS === '' ? '' :
            (window.TBG.TBGWidgets.validateSeedValue(rawS) ? window.TBG.TBGWidgets.formatSeedValue(rawS) : '');
        const normC = rawC === '' ? '' :
            (window.TBG.TBGWidgets.validateDenoiseValue(rawC) ? window.TBG.TBGWidgets.formatDenoiseValue(rawC) : '');

        setNodePropertySafe(node, propKeyDenoise(idx), normD);
        setNodePropertySafe(node, propKeySeed(idx), normS);
        setNodePropertySafe(node, propKeyCNet(idx), normC);

        msg.denoises[idx] = normD;
        msg.seeds[idx] = normS;
        msg.cnet_strength[idx] = normC;

        if (denoiseInput) denoiseInput.value = normD;
        if (seedInput) seedInput.value = normS;
        if (cnetInput) cnetInput.value = normC;
    });

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

        const prompts = msg.prompts || [];
        const denoises = msg.denoises || [];
        const seeds = msg.seeds || [];
        const cnet = msg.cnet_strength || [];

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

        msg.denoises = normDenoises;
        msg.seeds = normSeeds;
        msg.cnet_strength = normCnet;

        const hasContent = [...prompts, ...normDenoises, ...normSeeds, ...normCnet]
            .some(v => v != null && v !== '');

        const packed = JSON.stringify({
            prompts,
            denoises: normDenoises,
            seeds: normSeeds,
            cnet_strength: normCnet,
        });

        setNodePropertySafe(node, "tile_edits_json", packed);

        if (hasContent || force) {
            try {
                await fetch(api.apiURL("/TBG/McBoaty/v5/set_tile_edits_json"), {
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

    WRAPPER: (key, index, prompt, tile, denoise, seed, cnet_strength, node) => {
        const inputEl = document.createElement("div");
        inputEl.className = "comfy-wrapper-tgb";

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

            if (msg.seeds[index] !== v) {
                msg.seeds[index] = v;
                setNodePropertySafe(node, propKeySeed(index), v);
                const nodeWidget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
                TBGUpscalerTileNodeWidget.setValue(node, 'requeue', ((nodeWidget?.value) ?? 0) + 1);
                packEditsToJson(node);
            }
        };
        seedInput.addEventListener('input', debounce(persistSeed, 1000));
        seedInput.addEventListener('focusout', persistSeed);

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
                const nodeWidget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
                TBGUpscalerTileNodeWidget.setValue(node, 'requeue', ((nodeWidget?.value) ?? 0) + 1);
                packEditsToJson(node);
            }
        };
        cnet_strengthInput.addEventListener('input', debounce(persistCnet, 1000));
        cnet_strengthInput.addEventListener('focusout', persistCnet);

        const textarea = document.createElement("textarea");
        textarea.style.opacity = 0.6;
        textarea.style.flexGrow = 1;
        textarea.style.height = "100%";
        textarea.className = "comfy-multiline-input";
        textarea.value = prompt || "";
        textarea.placeholder = "tile " + text.textContent;
        textarea.dataId = "tile " + index;
        textarea.dataNodeId = node.id;

        const persistPrompt = () => {
            const msg = getMsg(node);
            const v = textarea.value.trim();
            if (msg.prompts[index] !== v) {
                msg.prompts[index] = v;
                setNodePropertySafe(node, propKeyPrompt(index), v);
                const nodeWidget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
                TBGUpscalerTileNodeWidget.setValue(node, 'requeue', ((nodeWidget?.value) ?? 0) + 1);
                packEditsToJson(node);
            }
        };
        textarea.addEventListener('input', debounce(persistPrompt, 1000));
        textarea.addEventListener('focusout', persistPrompt);

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
                const nodeWidget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
                TBGUpscalerTileNodeWidget.setValue(node, 'requeue', ((nodeWidget?.value) ?? 0) + 1);
                packEditsToJson(node);
            }
        };
        denoiseInput.addEventListener('input', debounce(persistDenoise, 1000));
        denoiseInput.addEventListener('focusout', persistDenoise);

        const img = document.createElement('img');
        img.src = imageDataToUrl(tile);

        const row_height = "140px";
        const inner_height = "140px";

        wrapper.style.display = "flex";
        wrapper.style.alignItems = "center";
        wrapper.style.gap = "12px";
        wrapper.style.width = "100%";
        wrapper.style.minHeight = row_height;
        wrapper.style.maxHeight = row_height;
        wrapper.style.overflow = "hidden";

        img.style.height = inner_height;
        img.style.width = inner_height;
        img.style.maxHeight = inner_height;
        img.style.maxWidth = inner_height;
        img.style.objectFit = "contain";
        img.style.flexShrink = "0";

        textarea.style.height = inner_height;
        textarea.style.minHeight = inner_height;
        textarea.style.maxHeight = inner_height;
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

        seedInput.style.height = "40px";
        seedInput.style.maxWidth = "100%";
        seedInput.style.boxSizing = "border-box";
        seedInput.style.flexShrink = "0";

        cnet_strengthInput.style.height = "40px";
        cnet_strengthInput.style.maxWidth = "100%";
        cnet_strengthInput.style.boxSizing = "border-box";
        cnet_strengthInput.style.flexShrink = "0";

        const denoiseRow = document.createElement("div");
        denoiseRow.style.display = "flex";
        denoiseRow.style.flexDirection = "column";
        denoiseRow.appendChild(denoiseInput);

        const seedRow = document.createElement("div");
        seedRow.style.display = "flex";
        seedRow.style.flexDirection = "column";
        seedRow.appendChild(seedInput);

        const cnet_strengthRow = document.createElement("div");
        cnet_strengthRow.style.display = "flex";
        cnet_strengthRow.style.flexDirection = "column";
        cnet_strengthRow.appendChild(cnet_strengthInput);

        rightControls.appendChild(denoiseRow);
        rightControls.appendChild(seedRow);
        rightControls.appendChild(cnet_strengthRow);

        wrapper.appendChild(text);
        wrapper.appendChild(img);
        wrapper.appendChild(textarea);
        wrapper.appendChild(rightControls);

        inputEl.appendChild(wrapper);

        const widget = node.addDOMWidget(key, "customtext", inputEl, {
            getValue() { return inputEl.value; },
            setValue(v) { inputEl.value = v; },
        });
        widget.inputEl = inputEl;

        TBGUpscalerTileNodeWidget.setValue(node, widget.name, prompt);
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

        const MAX_CLEAR = 512;
        for (let i = 0; i < MAX_CLEAR; i++) {
            try {
                delete node.properties[propKeyPrompt(i)];
                delete node.properties[propKeyDenoise(i)];
                delete node.properties[propKeySeed(i)];
                delete node.properties[propKeyCNet(i)];
            } catch (_) {}
        }

        setNodePropertySafe(node, "tile_edits_json", "");

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

        // Remove legacy "tile N" text widgets and any previous customtext wrappers
        node.widgets = node.widgets.filter(widget => {
            if (widget.type === "text" && /^tile\s+\d+$/i.test(widget.name)) {
                widget.onRemove?.();
                return false;
            }
            if (widget.type === "customtext") {
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
        this.setCnetStrengthInput(node);

        try {
            node.onResize?.(node.computeSize ? node.computeSize() : node.size);
            node.graph.setDirtyCanvas(true, true);
        } catch (_) {}

        // Tile rows
        this.setPrompterInputs(node);

        // Reset toggle only if rows exist
        if (msg.prompts.length) this.setCleanSwitch(node);

        // Height calc
        try {
            const tileCount = Array.isArray(msg.tiles) ? msg.tiles.length : 0;
            const H_ROW = 170, H_EXTRA = 180, H_MIN = 400, H_MAX = 20000;
            const total = H_EXTRA + tileCount * H_ROW;
            const height = Math.max(H_MIN, Math.min(H_MAX, total));
            if (Array.isArray(node.size)) node.size[1] = height;
            else if (node.size && typeof node.size === "object") node.size[1] = height;
            node.onResize?.(node.computeSize ? node.computeSize() : node.size);
            node.graph.setDirtyCanvas(true, true);
        } catch (_) {}

        // Hide stray legacy "tile N" widgets
        try {
            for (const w of node.widgets) {
                if (w.type === "text" && /^tile\s+\d+$/i.test(w.name || "")) {
                    w.hidden = true;
                    if (w.options) w.options.disabled = true;
                }
            }
            node.onResize?.(node.computeSize ? node.computeSize() : node.size);
            node.graph.setDirtyCanvas(true, true);
        } catch (_) {}
    }

    static init(node) {
        this.setIndexInput(node);
        this.setPromptInput(node);
        this.setDenoiseInput(node);
        this.setSeedInput(node);
        this.setCnetStrengthInput(node);
        this.setCleanSwitch(node);

        if (!node.__tbg_persist_hook_added__) {
            node.__tbg_persist_hook_added__ = true;

            api.addEventListener?.("queuePrompt", () => {
                try {
                    syncAllWrappersToProperties(node);
                    packEditsToJson(node);
                } catch (_) {}
            });
        }
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
                        if (widget.type === "customtext") {
                            const textarea = widget.inputEl.querySelector('textarea[placeholder^="tile "]');
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
                                    packEditsToJson(node);
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
                            if (widget.type === "customtext") {
                                const input = widget.inputEl.querySelector('input[placeholder^="denoise "]');
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

    static setCnetStrengthInput(node) {
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
                            if (widget.type === "customtext") {
                                const input = widget.inputEl.querySelector('input[placeholder^="cnet_strength "]');
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
                        if (widget.type === "customtext") {
                            const seedInput = widget.inputEl.querySelector('input[placeholder^="seed "]');
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

        node.widgets = node.widgets.filter(widget => {
            if (widget.type === "customtext") {
                try { widget.onRemove?.(); } catch (_) {}
                return false;
            }
            return true;
        });

        const count = Array.isArray(msg.tiles) ? msg.tiles.length : 0;

        for (let i = 0; i < count; i++) {
            const promptVal = getNodeProperty(node, `prompt_${i}`, "");
            const denoiseVal = getNodeProperty(node, `denoise_${i}`, "");
            const seedVal = getNodeProperty(node, `seed_${i}`, "");
            const cnetVal = getNodeProperty(node, `cnet_strength_${i}`, "");

            msg.prompts[i] = promptVal;
            msg.denoises[i] = denoiseVal;
            msg.seeds[i] = seedVal;
            msg.cnet_strength[i] = cnetVal;

            window.TBG.TBGWidgets.WRAPPER(
                `tile_${i}`,
                i,
                promptVal,
                msg.tiles[i],
                denoiseVal,
                seedVal,
                cnetVal,
                node
            );
        }

        packEditsToJson(node);
    }
}

/*──────────────────────────────────────────────────────────────────────────*/
/* Status overlay + extension registration                                  */
/*──────────────────────────────────────────────────────────────────────────*/

class TBGETUR {
    constructor() {
        if (!window.__TBGETUR__) {
            window.__TBGETUR__ = Symbol("__TBGETUR__");
        }
        this.symbol = window.__TBGETUR__;
    }

    getState(node) {
        return node[this.symbol] || {};
    }
    setState(node, state) {
        node[this.symbol] = state;
        app.canvas.setDirty(true);
    }

    addStatusTagHandler(nodeType) {
        if (nodeType[this.symbol]?.statusTagHandler) return;
        if (!nodeType[this.symbol]) nodeType[this.symbol] = {};
        nodeType[this.symbol] = { statusTagHandler: true };

        api.addEventListener("TBG/TBGETUR/update_status", ({ detail }) => {
            let { node, progress, text } = detail;
            const n = app.graph.getNodeById(+(node || app.runningNodeId));
            if (!n) return;

            const state = this.getState(n);
            state.status = Object.assign(state.status || {}, {
                progress: text ? progress : null,
                text: text || null,
            });
            this.setState(n, state);
        });

        const self = this;
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            const r = onDrawForeground?.apply?.(this, arguments);
            const state = self.getState(this);
            if (!state?.status?.text) return r;

            const { fgColor, bgColor, text, progress, progressColor } = { ...state.status };
            ctx.save();
            ctx.font = "12px sans-serif";
            const sz = ctx.measureText(text);
            ctx.fillStyle = bgColor || "dodgerblue";
            ctx.beginPath();
            ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, sz.width + 12, 20, 5);
            ctx.fill();

            if (progress) {
                ctx.fillStyle = progressColor || "green";
                ctx.beginPath();
                ctx.roundRect(0, -LiteGraph.NODE_TITLE_HEIGHT - 20, (sz.width + 12) * progress, 20, 5);
                ctx.fill();
            }

            ctx.fillStyle = fgColor || "#fff";
            ctx.fillText(text, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6);
            ctx.restore();
            return r;
        };
    }
}

const TBG_ETUR = new TBGETUR();

const myExtension = {
    name: "ComfyUI.TBG.TBGETUR",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        TBG_ETUR.addStatusTagHandler(nodeType);

        if (nodeData.name === "TBG_TilePrompter_v1") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                const r = onExecuted?.apply(this, arguments);

                const wrappers = this.widgets
                    .filter(w => w.type === "customtext" && w.inputEl)
                    .map(w => w.inputEl);

                const prompts = [];
                const denoises = [];
                const seeds = [];
                const cnet = [];
                const msg = getMsg(this);

                wrappers.forEach(wrapperEl => {
                    const textarea = wrapperEl.querySelector('textarea[placeholder^="tile "]');
                    if (!textarea) return;
                    const idx = parseInt(textarea.placeholder.replace(/\D/g, ""), 10) - 1;

                    const denInput = wrapperEl.querySelector('input[placeholder^="denoise "]');
                    const seedInput = wrapperEl.querySelector('input[placeholder^="seed "]');
                    const cnetInput = wrapperEl.querySelector('input[placeholder^="cnet_strength "]');

                    prompts[idx] = textarea.value || "";
                    denoises[idx] = denInput?.value || "";
                    seeds[idx] = seedInput?.value || "";
                    cnet[idx] = cnetInput?.value || "";

                    setNodePropertySafe(this, `prompt_${idx}`, (prompts[idx] || "").trim());
                    setNodePropertySafe(this, `denoise_${idx}`, (denoises[idx] || "").trim());
                    setNodePropertySafe(this, `seed_${idx}`, (seeds[idx] || "").trim());
                    setNodePropertySafe(this, `cnet_strength_${idx}`, (cnet[idx] || "").trim());
                });

                (async () => {
                    try {
                        const res = await fetch(`/TBG/McBoaty/v5/get_input_prompts?node=${this.id}`);
                        if (!res.ok) return;
                        const {prompts_in = []} = await res.json();

                        prompts_in.forEach((p, i) => {
                            if (p != "") {
                                setNodePropertySafe(this, `prompt_${i}`, p);
                                msg.prompts[i] = p;
                            }
                        });
                        packEditsToJson(this);
                        TBGUpscalerTileNodeWidget.refresh(this);
                    } catch (err) {
                        console.error("prompt fetch failed:", err);
                    }
                })();

                msg.prompts = prompts;
                msg.denoises = denoises;
                msg.seeds = seeds;
                msg.cnet_strength = cnet;
                msg.tiles = message.tiles || [];

                TBGUpscalerTileNodeWidget.refresh(this);
                packEditsToJson(this);
                return r;
            };

            if (!window.TBG._loadGraphHooked) {
                const origLoadGraphData = app.loadGraphData;
                app.loadGraphData = async function(data, ...args) {
                    const r = await origLoadGraphData.apply(this, [data, ...args]);

                    for (const n of app.graph._nodes) {
                        if (n.type === "TBG_TilePrompter_v1") {
                            try {
                                syncPropertiesToMessage(n);
                                packEditsToJson(n);

                                syncPropertiesToMessage(n);

                                {
                                  const m = getMsg(n);
                                  if (!Array.isArray(m.tiles)) m.tiles = [];
                                }
                                packEditsToJson(n);
                                TBGUpscalerTileNodeWidget.refresh(n);
                                packEditsToJson(n);
                                syncPropertiesToMessage(n);

                                (async () => {
                                    const nodeType = LiteGraph.registered_node_types["TBG_TilePrompter_v1"];
                                    const nodeData = { name: "TBG_TilePrompter_v1" };
                                    await myExtension.beforeRegisterNodeDef(nodeType, nodeData, app);
                                })();
                            } catch(e) {
                                console.warn("Initial sync failed for node", n.id, e);
                            }
                        }
                    }
                    return r;
                };
                window.TBG._loadGraphHooked = true;
            }

            if (!window.TBG._onNodeCreated) {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function() {

                    const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                    this.widgets = this.widgets.filter((widget) => {
                        if (widget.type === "text" && widget.name?.startsWith?.("tile ")) {
                            widget.onRemove?.();
                            return false;
                        }
                        if (widget.type === "customtext") {
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
                        const actualTileCount = Array.isArray(msg.tiles) ? msg.tiles.length : 0;

                        for (let i = 0; i < actualTileCount; i++) {
                            prompts[i] = getNodeProperty(this, propKeyPrompt(i), '');
                            denoises[i] = getNodeProperty(this, propKeyDenoise(i), '');
                            seeds[i] = getNodeProperty(this, propKeySeed(i), '');
                            cnet[i] = getNodeProperty(this, propKeyCNet(i), '');
                        }

                        clearExcessTileProperties(this, actualTileCount);

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

                        const inp = getInp(this);
                        inp.prompts = msg.prompts.slice();
                        inp.denoises = msg.denoises.slice();
                        inp.seeds = msg.seeds.slice();
                        inp.cnet_strength = msg.cnet_strength.slice();
                    } catch (_) {}

                    TBGUpscalerTileNodeWidget.init(this);
                    this.onResize?.(this.size);

                    setTimeout(() => {
                        try {
                            syncAllWrappersToProperties(this);
                            packEditsToJson(this);
                        } catch (e) {
                            console.warn("initial sync failed", e);
                        }
                    }, 10);

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
                                `/TBG/McBoaty/v5/tile_prompt?node=${this.id}&clientId=${api.clientId}&`
                            );
                            const res = await (await fetch(src)).json();
                            alert(res);
                        },
                    });
                }

                return r;
            };
        }
    },
};

app.registerExtension(myExtension);
