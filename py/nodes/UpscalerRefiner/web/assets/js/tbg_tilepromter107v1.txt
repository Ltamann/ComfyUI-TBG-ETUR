/*******************************************************************************************
 *
 *
 * PURPOSE
 * ───────
 * 1. Provide a "Tile Prompter" node that shows one row per image tile with:
 *    – prompt textarea
 *    – denoise value
 *    – seed
 *    – ControlNet strength
 *    – live tile preview
 * 2. Persist user edits to node.properties and a packed JSON string
 * 3. Offer header-level bulk inputs (apply same prompt/denoise/seed/CNet to many rows)
 * 4. Send status tags back from backend via custom event
 *
 * MAJOR SECTIONS
 * ──────────────
 * ① Global namespace + helpers
 * ② Widget creation + validation utilities
 * ③ Tile-row wrapper factory
 * ④ Widget manager (TBGUpscalerTileNodeWidget)
 * ⑤ JSON pack / sync helpers
 * ⑥ Reset handler
 * ⑦ Status-tag overlay
 * ⑧ Extension registration hook
 *
 * NOTE  all external names (TBG, TBGETUR, TBGUpscalerTileNodeWidget, etc.) are preserved.
 ******************************************************************************************/

 /*
 * McBoaty Tile-Prompter – rebuilt version for TGB enhanced upscaler and refiner pro
 *
 * Copyright (c) 2025 Tobias Laarmann
 * Copyright (c) 2024 David Asquiedge
 *
 * This file is a derivative of the original "McBoaty_v5.js" from
 * https://github.com/MaraScott/ComfyUI_MaraScott_Nodes. Copyright (c) 2024 David Asquiedge
 *
 * Released under the MIT License;
 *
 *  MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
**Attribution is required. The use of this software must be accompanied by proper
credit to the original author.**

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
// Attribution complies with the MIT licence by retaining this header.

*/
/*──────────────────────────────────────────────────────────────────────────*/
/* ① GLOBAL STATE & SIMPLE HELPERS                                         */
/*──────────────────────────────────────────────────────────────────────────*/

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";   // eslint-disable-line no-unused-vars

/*───────────────────────────────────────────────────────────────*/
/*  Republish tile_edits_json when backend restarts              */
/*───────────────────────────────────────────────────────────────*/
import { api } from "../../scripts/api.js";

// Listen for server connection status events
api.addEventListener("status", (event) => {
    console.log("[TBG_upscaler2] status event:", event.detail);
    if (event.detail.status === "disconnected") {
        console.warn("[TBG_upscaler2] Lost connection to server!");
    }
    if (event.detail.status === "connected") {
        console.log("[TBG_upscaler2] Reconnected to server.");
        // You can trigger a refresh of feature flags, metadata, etc. here
    }
});

// fires immediately when the socket drops
api.addEventListener('reconnecting', () => {
  console.warn('[TBG] backend went down – waiting…');
});

// fires many times; use the first non-null payload as "reconnected"

function resyncTilePrompters() {
  Object.values(app.graph._nodes).forEach((n) => {
    // Only check the internal type (stable)
    if (n?.type === "TBG_TilePrompter_v1") {
      try {
        // 1️⃣ Sync widget → node.properties → TBGETUR.message
        syncAllWrappersToProperties(n);

        // 2️⃣ packEditsToJson() will POST to your endpoint
        console.debug(
          `[TBG_upscaler2] Node ${n.id} (${n.type}): tile_edits_json republished after backend restart`
        );
      } catch (err) {
        console.error(`[TBG_upscaler2] Failed to resync node ${n.id}:`, err);
      }
    }
  });
}

let waitingForReconnect = false;

api.addEventListener("status", (event) => {
    console.log("[TBG_upscaler2] status event:", event.detail);
    
    // ✅ FIXED: Check if event.detail exists before accessing its properties
    if (!event.detail) {
        console.warn("[TBG_upscaler2] Status event fired with null detail");
        return; // Exit early if detail is null
    }
    
    if (event.detail.status === "disconnected") {
        console.warn("[TBG_upscaler2] Lost connection to server!");
    }
    if (event.detail.status === "connected") {
        console.log("[TBG_upscaler2] Reconnected to server.");
    }
});

/* Create one Global namespace TBG object if it does not exist */
if (!window.TBG) window.TBG = {};

/* Create TBGETUR (Tile-Based GET-Update-Render) namespace once */
if (!window.TBG.TBGETUR) {
    window.TBG.TBGETUR = {
        /* flags */
        init:  false,
        clean: false,

        /* params - preset option lists the widgets will use (initially empty). */
        params: {
            denoise:       { values: [] },
            seed:          { values: [] }, // reserved
            cnet_strength: { values: [] }  // reserved
        },

        /* runtime caches (message - arrays that travel to/from the Python back-end.) */
        message: {
            prompts:       [],
            tiles:         [],
            denoises:      [],
            seeds:         [],
            cnet_strength: []
        },

        /* original values loaded from backend graph JSON a shallow copy of the original graph values (useful for reset)*/
        inputs: {
            prompts:       [],
            tiles:         [],
            denoises:      [],
            seeds:         [],
            cnet_strength: []
        }
    };
}

/*********** Utility: property key builders ********************************/
/* Turns a numeric index into the property name stored on the LiteGraph node.
Used everywhere a tile value is read from / written to node.properties. */

function propKeyPrompt(i) {return `prompt_${i}`;}
function propKeyDenoise(i) {return `denoise_${i}`;}
function propKeySeed(i) {return `seed_${i}`;}
function propKeyCNet(i) {return `cnet_strength_${i}`;}

/*********** Utility: safely set/get node.properties ***********************/

/* Called by every persist or bulk-apply helper. */
function setNodePropertySafe(node, key, value) {
    try {
        node.setProperty(key, value);
        node.setDirtyCanvas(true);
    } catch (_) {/* ignore */ }
}

/* getNodeProperty – mirrors the above for reads with a fallback value.
Used by the initial sync that re-hydrates the UI from stored data. */
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

/*********** Utility: simple debounce after input *************************************/
function debounce(fn, delay = 1000) {
    //delay = 2000;
    let t;
    return (...args) => {
        clearTimeout(t);
        t = setTimeout(() => fn(...args), delay);
    };
}

/*********** ✅ FIXED: Clear excess properties helper *************************/
function clearExcessTileProperties(node, currentTileCount) {
    const MAX_CLEAR = 512;
    for (let i = currentTileCount; i < MAX_CLEAR; i++) {
        const hasAnyProperty = [
            getNodeProperty(node, `prompt_${i}`, null),
            getNodeProperty(node, `denoise_${i}`, null), 
            getNodeProperty(node, `seed_${i}`, null),
            getNodeProperty(node, `cnet_strength_${i}`, null)
        ].some(prop => prop !== null);
        
        if (!hasAnyProperty) break; // No more properties to clear
        
        try {
            delete node.properties[propKeyPrompt(i)];
            delete node.properties[propKeyDenoise(i)];
            delete node.properties[propKeySeed(i)];
            delete node.properties[propKeyCNet(i)];
        } catch (_) {}
    }
}

/*──────────────────────────────────────────────────────────────────────*/
/* ② VALIDATION + FORMAT HELPERS                                        */
/*──────────────────────────────────────────────────────────────────────*/

/* syncPropertiesToMessage
Scans node.properties (prompt_0, denoise_0, …)
Fills the in-memory arrays under window.TBG.TBGETUR.message.*.

Runs when:
 - The graph is loaded (so the UI shows the existing data).
 - onExecuted finishes (to capture changes made by Python).
 - The helper is also called from an ad-hoc debug console log.

✅ FIXED: On graph load, we need to scan ALL properties first to find what was saved,
then we can determine the actual tile count and clear excess properties.
*/

function syncPropertiesToMessage(node) {
    const prompts = [];
    const denoises = [];
    const seeds = [];
    const cnet = [];
    
    // ✅ FIXED: On graph load, we must scan ALL properties first to find saved data
    // We can't rely on tiles.length because it might be empty initially
    const MAX = 512;
    let actualTileCount = 0;

    for (let i = 0; i < MAX; i++) {
        const p = getNodeProperty(node, `prompt_${i}`, null);
        const d = getNodeProperty(node, `denoise_${i}`, null);
        const s = getNodeProperty(node, `seed_${i}`, null);
        const c = getNodeProperty(node, `cnet_strength_${i}`, null);
        
        // If we find ANY property for this index, count it
        if (p !== null || d !== null || s !== null || c !== null) {
            actualTileCount = i + 1;
            prompts[i] = p ?? "";
            denoises[i] = d ?? "";
            seeds[i] = s ?? "";
            cnet[i] = c ?? "";
        } else if (i > 0 && actualTileCount === 0) {
            // No properties found at all, break early
            break;
        }
    }

    // ✅ FIXED: Now clear excess properties beyond what we actually found
    clearExcessTileProperties(node, actualTileCount);

    window.TBG.TBGETUR.message.prompts = prompts;
    window.TBG.TBGETUR.message.denoises = denoises;
    window.TBG.TBGETUR.message.seeds = seeds;
    window.TBG.TBGETUR.message.cnet_strength = cnet;
    
    // ✅ FIXED: Ensure tiles array matches the properties we found
    // If we have saved properties but no tiles array, create placeholder tiles
    if (actualTileCount > 0 && (!Array.isArray(window.TBG.TBGETUR.message.tiles) || window.TBG.TBGETUR.message.tiles.length < actualTileCount)) {
        const existingTiles = Array.isArray(window.TBG.TBGETUR.message.tiles) ? window.TBG.TBGETUR.message.tiles : [];
        window.TBG.TBGETUR.message.tiles = Array.from({ length: actualTileCount }, (_, i) => {
            return existingTiles[i] || { filename: "", type: "", subfolder: "" };
        });
    }
}

// Persist all current wrapper inputs into node.properties and message arrays
/*
Walks every custom tile wrapper in the UI, validates / normalises user input, and writes it back to both:
 - node.properties – persistent storage in the graph.
 - window.TBG.TBGETUR.message.* – the live cache sent to back-end.
Runs:
 - When the user edits a field (debounced).
 - Right before a queue prompt is sent (api.addEventListener("queuePrompt",…)). */

function syncAllWrappersToProperties(node) {
    const wrappers = (node.widgets || [])
        .filter(w => w.type === "customtext" && w.inputEl)
        .map(w => w.inputEl);

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

        window.TBG.TBGETUR.message.denoises[idx] = normD;
        window.TBG.TBGETUR.message.seeds[idx] = normS;
        window.TBG.TBGETUR.message.cnet_strength[idx] = normC;

        if (denoiseInput) denoiseInput.value = normD; // mirror corrected value
        if (seedInput) seedInput.value = normS; // mirror corrected value
        if (cnetInput) cnetInput.value = normC; // mirror corrected value
    });

    packEditsToJson(node);
}

/* packEditsToJson Builds tile_edits_json – a compact JSON of all four arrays.

- Stores it on the node (setNodePropertySafe).
- If at least one entry is non-empty, POSTs it to /TBG/McBoaty/v5/set_tile_edits_json so the Python side can read it.

Runs after every local change and after graph load to keep browser ↔ server in sync.
 */

async function packEditsToJson(node, force = true) {

    try {
        if (!node) {
            console.warn("packEditsToJson called without a node");
            return false;
        }
        // Read arrays once
        const prompts = window.TBG.TBGETUR.message.prompts || [];
        const denoises = window.TBG.TBGETUR.message.denoises || [];
        const seeds = window.TBG.TBGETUR.message.seeds || [];
        const cnet = window.TBG.TBGETUR.message.cnet_strength || [];

        // Normalize copies
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

        // Optionally mirror back normalized arrays to message.*
        window.TBG.TBGETUR.message.denoises = normDenoises;
        window.TBG.TBGETUR.message.seeds = normSeeds;
        window.TBG.TBGETUR.message.cnet_strength = normCnet;

        // Determine if there is any meaningful content
        const hasContent = [...prompts, ...normDenoises, ...normSeeds, ...normCnet]
            .some(v => v != null && v !== '');

        // Build packed from normalized arrays
        const packed = JSON.stringify({
            prompts,
            denoises: normDenoises,
            seeds: normSeeds,
            cnet_strength: normCnet,
        });

        // Persist locally
        setNodePropertySafe(node, "tile_edits_json", packed);

        // Mirror to server only when we have content
        if (hasContent || force) {
            try {
                await fetch(api.apiURL("/TBG/McBoaty/v5/set_tile_edits_json"), {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        node: String(node.id),
                        tile_edits_json: packed
                    }),
                });
            } catch (e) {
                console.warn("mirror tile_edits_json failed:", e);
            }
        } else {
            //console.debug("skipped mirroring empty tile_edits_json for node", node.id);
        }

        //console.debug("tile_edits_json set on node", node.id, packed);
        return true;
    } catch (e) {
        console.warn("packEditsToJson failed:", e);
        return false;
    }
}

// UI Widgets for each tile row TBGWidgets utility object
/* Holds validators / formatters for:

- Denoise, Cnet (0 – 1, 2 dp)
- Seed (integer string)

Plus WRAPPER – the factory that builds one tile row:
 - <p> index label
 - <img> preview
 - <textarea> prompt
 - Three <input>s (denoise, seed, cnet_strength)
 - Right-hand column container

Each field registers input and focusout handlers that:
 - normalise the value,
 - update message.*,
 - write to node.properties,
 - call packEditsToJson,
 - "bump" the hidden requeue widget so the graph re-runs.

Called by the widget manager whenever rows need to (re)render. */

window.TBG.TBGWidgets = {
    // Denoise validation and formatting (0.00–1.00)
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

    // Seed must be integer or empty
    validateSeedValue: (value) => {
        // Always "valid" after sanitization approach
        return true;
    },

    formatSeedValue: (value) => {
        // Sanitize by stripping non-digits; optionally preserve a single leading '-'
        let s = String(value ?? '').trim();

        // Keep leading '-' if present (remove this block if you don't want negatives)
        const isNeg = s.startsWith('-');
        if (isNeg) s = s.slice(1);

        // Remove all non-digit characters
        s = s.replace(/\D+/g, '');

        // Restore '-' only if digits remain
        if (isNeg && s.length > 0) s = '-' + s;

        return s;
    },

    // Per-tile UI row (prompt, denoise, seed, cnet_strength + preview)
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

        // Seed input
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
            let v = seedInput.value.trim();
            if (!window.TBG.TBGWidgets.validateSeedValue(v)) v = '';
            else v = window.TBG.TBGWidgets.formatSeedValue(v);
            // NEW: write normalized value back to the UI immediately
            if (seedInput.value !== v) seedInput.value = v;

            if (window.TBG.TBGETUR.message.seeds[index] !== v) {
                window.TBG.TBGETUR.message.seeds[index] = v;
                setNodePropertySafe(node, propKeySeed(index), v);
                const nodeWidget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
                TBGUpscalerTileNodeWidget.setValue(node, 'requeue', ((nodeWidget?.value) ?? 0) + 1);
                packEditsToJson(node);
            }

        };
        seedInput.addEventListener('input', debounce(persistSeed, 1000));
        seedInput.addEventListener('focusout', persistSeed);

        // cnet_strength input (0.00–1.00 similar to denoise)
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
            let v = cnet_strengthInput.value.trim();
            if (!window.TBG.TBGWidgets.validateDenoiseValue(v)) v = '';
            else v = window.TBG.TBGWidgets.formatDenoiseValue(v);

            if (cnet_strengthInput.value !== v) cnet_strengthInput.value = v;
            if (window.TBG.TBGETUR.message.cnet_strength[index] !== v) {
                window.TBG.TBGETUR.message.cnet_strength[index] = v;
                setNodePropertySafe(node, propKeyCNet(index), v);
                const nodeWidget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
                TBGUpscalerTileNodeWidget.setValue(node, 'requeue', ((nodeWidget?.value) ?? 0) + 1);
                packEditsToJson(node);
            }
        };
        cnet_strengthInput.addEventListener('input', debounce(persistCnet, 1000));
        cnet_strengthInput.addEventListener('focusout', persistCnet);

        // Prompt textarea
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
            const v = textarea.value.trim();

            if (window.TBG.TBGETUR.message.prompts[index] !== v) {
                window.TBG.TBGETUR.message.prompts[index] = v;
                setNodePropertySafe(node, propKeyPrompt(index), v);
                const nodeWidget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
                TBGUpscalerTileNodeWidget.setValue(node, 'requeue', ((nodeWidget?.value) ?? 0) + 1);
                packEditsToJson(node);
            }
        };
        textarea.addEventListener('input', debounce(persistPrompt, 1000));
        textarea.addEventListener('focusout', persistPrompt);

        // Denoise input
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
            let v = denoiseInput.value.trim();
            if (!window.TBG.TBGWidgets.validateDenoiseValue(v)) v = '';
            else v = window.TBG.TBGWidgets.formatDenoiseValue(v);
            if (denoiseInput.value !== v) denoiseInput.value = v;
            if (window.TBG.TBGETUR.message.denoises[index] !== v) {
                window.TBG.TBGETUR.message.denoises[index] = v;
                setNodePropertySafe(node, propKeyDenoise(index), v);
                const nodeWidget = TBGUpscalerTileNodeWidget.getByName(node, 'requeue');
                TBGUpscalerTileNodeWidget.setValue(node, 'requeue', ((nodeWidget?.value) ?? 0) + 1);
                packEditsToJson(node);
            }
        };
        denoiseInput.addEventListener('input', debounce(persistDenoise, 1000));
        denoiseInput.addEventListener('focusout', persistDenoise);

        // Image preview
        const img = document.createElement('img');
        img.src = imageDataToUrl(tile);
        //img.alt = prompt;

        // Layout metrics
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

        // Right-side vertical controls
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
            getValue() {
                return inputEl.value;
            },
            setValue(v) {
                inputEl.value = v;
            },
        });
        widget.inputEl = inputEl;

        // keep prompt value reference
        TBGUpscalerTileNodeWidget.setValue(node, widget.name, prompt);

        return widget;
    },
};

// Reset/clean handler
/*
Implements the Reset toggle:

 - Clears all message arrays and node.properties for every tile.
 - Calls packEditsToJson (empty payload ⇒ nothing posted).
 - Refreshes the widget layout so the UI collapses.

Runs when the user clicks the "Reset" switch at the top of the node. */

class TBG_NodePrompter {
    static async clean(node) {
        window.TBG.TBGETUR.clean = false;
        const cleanedLabel = "";
        const nodeTitle = node.title;
        node.title = nodeTitle + cleanedLabel;

        // Reset arrays and clear node properties
        window.TBG.TBGETUR.inputs.tiles = [];
        window.TBG.TBGETUR.message.tiles = [];
        window.TBG.TBGETUR.message.prompts = [];
        window.TBG.TBGETUR.inputs.prompts = [];
        window.TBG.TBGETUR.inputs.denoises = [];
        window.TBG.TBGETUR.message.denoises = [];
        window.TBG.TBGETUR.inputs.seeds = [];
        window.TBG.TBGETUR.message.seeds = [];
        window.TBG.TBGETUR.inputs.cnet_strength = [];
        window.TBG.TBGETUR.message.cnet_strength = [];

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

        setTimeout(() => {
            node.title = nodeTitle;
        }, 500);
    }
}

/*
TBGUpscalerTileNodeWidget (widget manager)

Central class that owns every LiteGraph widget of the node.

Key static helpers:
 - getByName / setValue – thin wrappers around LiteGraph widget list.
 - refresh(node) – full rebuild sequence
    - Strip legacy widgets.
    - Add header controls (Index filter, Prompt, etc.).
    - Add/refresh tile rows via setPrompterInputs.
    - Resize the node (height = 180 px + 170 px × row count).

 - _addText / _addCombo – create header widgets; _addCombo fills its values array from window.TBG.TBGETUR.params.*.
 - setPrompterInputs – loops tiles.length times, reads any persisted values, then calls TBGWidgets.WRAPPER to build each row.
 - setCleanSwitch – adds the Reset toggle when at least one row exists.

When is refresh() executed?
 - Node first created (init)
 - Graph load re-hydration
 - After back-end execution (onExecuted)
 - Any time the user changes the Index filter or bulk widgets. */

// Widget manager
class TBGUpscalerTileNodeWidget {
    static INDEX = {
        name: "Filter by Indexes",
        default: ""
    };
    static PROMPT = {
        name: "Prompt",
        default: ""
    };
    static DENOISE = {
        name: "Denoise",
        values: ['unchanged', 'Use Global Denoise'],
        default: "unchanged",
        min: 0.00,
        max: 1.00,
        step: 0.01,
    };
    static SEED = {
        name: "Seed",
        default: ""
    };
    static CNET_STRENGTH = {
        name: "CNet Strength",
        values: ['unchanged', 'Use Global CNet Strength'],
        default: "unchanged",
        min: 0.00,
        max: 1.00,
        step: 0.01,
    };
    static CLEAN = {
        name: 'Reset',
        default: false
    };

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
        // 1) Remove legacy "tile N" text widgets and any previous customtext wrappers
        node.widgets = node.widgets.filter(widget => {
            if (widget.type === "text" && /^tile\s+\d+$/i.test(widget.name)) {
                widget.onRemove?.();
                return false;
            }
            if (widget.type === "customtext") {
                try {
                    widget.onRemove?.();
                } catch (_) {}
                return false;
            }
            return true;
        });

        // 2) Header controls
        this.setIndexInput(node);
        this.setPromptInput(node);
        this.setDenoiseInput(node);
        this.setSeedInput(node);
        this.setCnetStrengthInput(node);

        // 3) Baseline reflow
        try {
            node.onResize?.(node.computeSize ? node.computeSize() : node.size);
            node.graph.setDirtyCanvas(true, true);
        } catch (_) {}

        // 4) Tile rows
        this.setPrompterInputs(node);

        // 5) Reset toggle only if rows exist
        if (window.TBG.TBGETUR.message.prompts.length) this.setCleanSwitch(node);

        // 6) Height calc
        try {
            const tileCount = Array.isArray(window.TBG.TBGETUR.message.tiles) ?
                window.TBG.TBGETUR.message.tiles.length :
                0;
            const H_ROW = 170,
                H_EXTRA = 180,
                H_MIN = 400,
                H_MAX = 20000;
            const total = H_EXTRA + tileCount * H_ROW;
            const height = Math.max(H_MIN, Math.min(H_MAX, total));
            if (Array.isArray(node.size)) node.size[1] = height;
            else if (node.size && typeof node.size === "object") node.size[1] = height;
            node.onResize?.(node.computeSize ? node.computeSize() : node.size);
            node.graph.setDirtyCanvas(true, true);
        } catch (_) {}

        // 7) Hide stray legacy "tile N" widgets
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

        // Pre-run sync even if no blur happened
        if (!node.__tbg_persist_hook_added__) {
            node.__tbg_persist_hook_added__ = true;

            // Hook queue event
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
                                    window.TBG.TBGETUR.message.prompts[realIndexValue] = v;
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
                                        input.value = v; // keep corrected value visible
                                        setNodePropertySafe(node, propKeyDenoise(realIndexValue), v);
                                        window.TBG.TBGETUR.message.denoises[realIndexValue] = v;
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
                                        input.value = v; // keep corrected value visible
                                        setNodePropertySafe(node, propKeyCNet(realIndexValue), v);
                                        window.TBG.TBGETUR.message.cnet_strength[realIndexValue] = v;
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
                                    window.TBG.TBGETUR.message.seeds[realIndexValue] = v;
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
        // Remove existing customtext wrappers to prevent duplicates
        node.widgets = node.widgets.filter(widget => {
            if (widget.type === "customtext") {
                try {
                    widget.onRemove?.();
                } catch (_) {}
                return false;
            }
            return true;
        });

        // Build rows
        const count = Array.isArray(window.TBG.TBGETUR.message.tiles) ?
            window.TBG.TBGETUR.message.tiles.length :
            0;

        for (let i = 0; i < count; i++) {
            const promptVal = getNodeProperty(node, `prompt_${i}`, "");
            const denoiseVal = getNodeProperty(node, `denoise_${i}`, "");
            const seedVal = getNodeProperty(node, `seed_${i}`, "");
            const cnetVal = getNodeProperty(node, `cnet_strength_${i}`, "");

            // Mirror persisted values into message arrays so first run has content
            window.TBG.TBGETUR.message.prompts[i] = promptVal;
            window.TBG.TBGETUR.message.denoises[i] = denoiseVal;
            window.TBG.TBGETUR.message.seeds[i] = seedVal;
            window.TBG.TBGETUR.message.cnet_strength[i] = cnetVal;

            // Create wrapper
            window.TBG.TBGWidgets.WRAPPER(
                `tile_${i}`,
                i,
                promptVal,
                window.TBG.TBGETUR.message.tiles[i],
                denoiseVal,
                seedVal,
                cnetVal,
                node
            );
        }

        // Pack once after all rows created
        packEditsToJson(node);
    }
}

/*
11. TBGETUR_Status (status-tag overlay)
Adds a tiny progress bar under the node title:
 - Listens for the custom DOM event TBG/TBGETUR/update_status that the Python side emits.
 - Stores {text, progress} into a hidden symbol on the node.
 - Patches onDrawForeground to paint a rounded rectangle and progress fill.

Always active for every node type once the extension registers. */

// Status overlay and Extension registration
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
        nodeType[this.symbol] = {
            statusTagHandler: true
        };

        api.addEventListener("TBG/TBGETUR/update_status", ({
            detail
        }) => {
            let {
                node,
                progress,
                text
            } = detail;
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

            const {
                fgColor,
                bgColor,
                text,
                progress,
                progressColor
            } = {
                ...state.status
            };
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

                // Collect post-run values (for UI consistency next time)
                const wrappers = this.widgets
                    .filter(w => w.type === "customtext" && w.inputEl)
                    .map(w => w.inputEl);

                const prompts = [];
                const denoises = [];
                const seeds = [];
                const cnet = [];

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

                    // persist to properties for next run
                    setNodePropertySafe(this, `prompt_${idx}`, (prompts[idx] || "").trim());
                    setNodePropertySafe(this, `denoise_${idx}`, (denoises[idx] || "").trim());
                    setNodePropertySafe(this, `seed_${idx}`, (seeds[idx] || "").trim());
                    setNodePropertySafe(this, `cnet_strength_${idx}`, (cnet[idx] || "").trim());
                });

                    // get promts from node py
                    const nodeId = this.id;

                    (async () => {
                        try {
                          const res = await fetch(`/TBG/McBoaty/v5/get_input_prompts?node=${this.id}`);
                          if (!res.ok) return;
                          const {prompts_in = []} = await res.json();

                          prompts_in.forEach((p, i) => {
                          if (p != "") {
                            setNodePropertySafe(this, `prompt_${i}`, p);
                            window.TBG.TBGETUR.message.prompts[i] = p;
                            }
                          });
                          packEditsToJson(this);               // run after async work
                          TBGUpscalerTileNodeWidget.refresh(this);
                        } catch (err) {
                          console.error("prompt fetch failed:", err);
                        }
                      })();

                window.TBG.TBGETUR.message.prompts = prompts;
                window.TBG.TBGETUR.message.denoises = denoises;
                window.TBG.TBGETUR.message.seeds = seeds;
                window.TBG.TBGETUR.message.cnet_strength = cnet;
                window.TBG.TBGETUR.message.tiles = message.tiles || [];

                // Use this (the node instance), not node
                //packEditsToJson(this);

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
                                syncPropertiesToMessage(n);   // <── read values from properties
                                packEditsToJson(n);           // <── push into JSON + cache
                                //console.debug("Initial sync (props) done for node", n.id);

                                syncPropertiesToMessage(n);            // fills message.prompts/denoises/seeds/cnet_strength from properties

                                // ✅ FIXED: Don't expand tiles beyond what's actually provided by backend
                                {
                                  const m = window.TBG.TBGETUR.message;
                                  // Don't expand tiles array - keep only what's actually provided
                                  if (!Array.isArray(m.tiles)) {
                                    m.tiles = [];
                                  }
                                  // No expansion logic - let actual tile count drive everything
                                }
                                packEditsToJson(n);
                                TBGUpscalerTileNodeWidget.refresh(n);
                                packEditsToJson(n);
                                syncPropertiesToMessage(n);

                                  (async () => {
                          // fake test inputs, or real ones if you have them
                          const nodeType = LiteGraph.registered_node_types["TBG_TilePrompter_v1"];
                            const nodeData = { name: "TBG_TilePrompter_v1" }; // minimal stub, or grab from backend

                          // `app` is your real ComfyUI app object
                          await myExtension.beforeRegisterNodeDef(nodeType, nodeData, app);

                        })();
                    } catch(e) {
                        console.warn("Initial sync failed for node", n.id, e);
                    }
                }
            }
            return r;
        };
 window.TBG._loadGraphHooked = true;   // ← prevents re-wrapping
}

if (!window.TBG._onNodeCreated) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {

                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Remove legacy text widgets and any pre-existing customtext wrappers
                this.widgets = this.widgets.filter((widget) => {
                    if (widget.type === "text" && widget.name?.startsWith?.("tile ")) {
                        widget.onRemove?.();
                        return false;
                    }
                    if (widget.type === "customtext") {
                        try {
                            widget.onRemove?.();
                        } catch (_) {}
                        return false;
                    }
                    return true;
                });

                // ✅ FIXED: Rehydrate from node.properties - only scan actual tiles
                try {
                    // ✅ FIXED: Get actual tile count first, don't probe 512
                    const actualTileCount = Array.isArray(window.TBG.TBGETUR.message.tiles) ? 
                        window.TBG.TBGETUR.message.tiles.length : 0;
                    
                    const prompts = [];
                    const denoises = [];
                    const seeds = [];
                    const cnet = [];

                    // ✅ FIXED: Only probe for actual tiles
                    for (let i = 0; i < actualTileCount; i++) {
                        prompts[i] = getNodeProperty(this, propKeyPrompt(i), '');
                        denoises[i] = getNodeProperty(this, propKeyDenoise(i), '');
                        seeds[i] = getNodeProperty(this, propKeySeed(i), '');
                        cnet[i] = getNodeProperty(this, propKeyCNet(i), '');
                    }

                    // Clear excess properties beyond actual count
                    clearExcessTileProperties(this, actualTileCount);

                    const mergePrefProp = (propArr, cacheArr = []) => {
                        const n = Math.max(propArr.length, cacheArr.length);
                        const out = new Array(n);
                        for (let k = 0; k < n; k++) {
                            out[k] = (propArr[k] !== undefined && propArr[k] !== null) ? propArr[k] : (cacheArr[k] ?? '');
                        }
                        return out;
                    };

                    window.TBG.TBGETUR.message.prompts = mergePrefProp(prompts, window.TBG.TBGETUR.message.prompts);
                    window.TBG.TBGETUR.message.denoises = mergePrefProp(denoises, window.TBG.TBGETUR.message.denoises);
                    window.TBG.TBGETUR.message.seeds = mergePrefProp(seeds, window.TBG.TBGETUR.message.seeds);
                    window.TBG.TBGETUR.message.cnet_strength = mergePrefProp(cnet, window.TBG.TBGETUR.message.cnet_strength);

                    window.TBG.TBGETUR.inputs.prompts = window.TBG.TBGETUR.message.prompts.slice();
                    window.TBG.TBGETUR.inputs.denoises = window.TBG.TBGETUR.message.denoises.slice();
                    window.TBG.TBGETUR.inputs.seeds = window.TBG.TBGETUR.message.seeds.slice();
                    window.TBG.TBGETUR.inputs.cnet_strength = window.TBG.TBGETUR.message.cnet_strength.slice();
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
                }, 10); // slight delay lets properties restore

                // At the end of onNodeCreated (inside the TBG TilePrompter_v1 branch):
                return r;
            };
 window.TBG._onNodeCreated = true;   // ← prevents re-wrapping
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
