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


    },
});
