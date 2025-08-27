import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "TBG.TiledUpscalerCE",
    
    init() {
        const STRING = ComfyWidgets.STRING;
        ComfyWidgets.STRING = function (node, inputName, inputData) {
            const r = STRING.apply(this, arguments);
            r.widget.dynamicPrompts = inputData?.[1].dynamicPrompts;
            return r;
        };
    },




    beforeRegisterNodeDef(nodeType) {
        if (nodeType.comfyClass === "TBG_Upscaler_v1_pro") {
            const onDrawForeground = nodeType.prototype.onDrawForeground;

            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawForeground?.apply?.(this, arguments);

                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v && v.value) {
                    const text = v.value[0] + "";
                    ctx.save();
                    ctx.font = "7px";
                    // ctx.fillStyle = "dodgerblue";
                    const sz = ctx.measureText(text);
                    
                    // Move text to header area - change Y position
                    // ctx.fillText(text, this.size[0] - sz.width - 5, -15);
                    ctx.fillText(text, 20, 80);
                    ctx.restore();
                }

                return r;
            };
        }

        if (nodeType.comfyClass === "TBG_magnific_ETUR") {
            const onDrawForeground = nodeType.prototype.onDrawForeground;

            nodeType.prototype.onDrawForeground = function (ctx) {
                const r = onDrawForeground?.apply?.(this, arguments);

                const v = app.nodeOutputs?.[this.id + ""];
                if (!this.flags.collapsed && v && v.value) {
                    const text = v.value[0] + "";
                    ctx.save();
                    // ctx.font = "6px";
                    // ctx.fillStyle = "dodgerblue";
                    const sz = ctx.measureText(text);

                    // Move text to header area - change Y position
                    // ctx.fillText(text, this.size[0] - sz.width - 5, -15);
                    ctx.fillText(text, 50, 218);
                    ctx.restore();
                }

                return r;
            };
        }

if (nodeType.comfyClass === "TBG_Refiner_v1_pro") {
    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated?.apply?.(this, arguments);

        // Container
        const container = document.createElement("div");
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.gap = "4px";
        container.style.padding = "4px";
        container.style.marginBottom = "8px"; // <-- added margin at the end

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
        seedField.style.background = "#444";      // ComfyUI gray
        seedField.style.color = "#eee";           // light text
        seedField.style.border = "1px solid #666";
        seedField.style.borderRadius = "999px";   // pill/half circle
        seedField.style.padding = "4px 8px";
        seedField.style.fontSize = "12px";
        
        // Copy button
        const copyBtn = document.createElement("button");
        copyBtn.innerText = "Copy";
        copyBtn.style.background = "#555";
        copyBtn.style.color = "#eee";
        copyBtn.style.border = "1px solid #666";
        copyBtn.style.borderRadius = "999px";     // pill style
        copyBtn.style.padding = "4px 10px";
        copyBtn.style.cursor = "pointer";
        copyBtn.style.fontSize = "12px";
       

        copyBtn.onmouseenter = () => copyBtn.style.background = "#666";
        copyBtn.onmouseleave = () => copyBtn.style.background = "#555";

        copyBtn.onclick = () => {
            navigator.clipboard.writeText(seedField.value);
        };


        // Assemble
        row.appendChild(seedField);
        row.appendChild(copyBtn);
    
        container.appendChild(row);
        container.appendChild(label);

        const spacer = document.createElement("div");
        spacer.style.height = "8px"; // adjust the spacing here
        container.appendChild(spacer);

        this.addDOMWidget("Seed", "div", container, { serialize: false });
        this.seedField = seedField;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
        onExecuted?.apply?.(this, arguments);

        if (output && output.value) {
            const text = output.value[0] + "";
            if (this.seedField) {
                this.seedField.value = text;
            }
        }
    };
}





    },
});
