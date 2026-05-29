import { app } from "../../../scripts/app.js";

export function resolveTBGNodeClass(nodeType, nodeData) {
  return String(
    nodeType?.comfyClass ||
      nodeData?.name ||
      nodeType?.type ||
      nodeType?.title ||
      ""
  );
}

export function isTBGNode(nodeType, nodeData, names) {
  const cls = resolveTBGNodeClass(nodeType, nodeData);
  return names.includes(cls) || names.includes(String(nodeData?.name || ""));
}

export function getNodeWidgets(node) {
  return Array.isArray(node?.widgets) ? node.widgets : [];
}

export function getNodeOutput(node) {
  return app?.nodeOutputs?.[String(node?.id ?? "")] ?? null;
}

export function getOutputValue(node, fallback) {
  const direct = fallback?.ui?.value ?? fallback?.value ?? null;
  if (Array.isArray(direct)) return direct;

  const output = getNodeOutput(node);
  const value = output?.value;
  if (Array.isArray(value)) return value;

  return null;
}

export function safeApply(fn, ctx, args = []) {
  if (typeof fn !== "function") return undefined;
  return fn.apply(ctx, args);
}

export function attachDOMWidget(node, name, element, options = {}) {
  if (typeof node?.addDOMWidget !== "function") return null;
  return node.addDOMWidget(name, "div", element, { serialize: false, ...options });
}

export function requestNodeRedraw(node) {
  try {
    node?.onResize?.(node.size);
  } catch (_) {}
  try {
    node?.setDirtyCanvas?.(true, true);
  } catch (_) {}
  try {
    node?.graph?.setDirtyCanvas?.(true, true);
  } catch (_) {}
}

export function setNodeMinHeight(node, minHeight) {
  const width = Array.isArray(node?.size) ? node.size[0] : node?.size?.[0];
  const currentHeight = Array.isArray(node?.size) ? node.size[1] : node?.size?.[1];
  const nextHeight = Math.max(Number(currentHeight ?? 0), Number(minHeight ?? 0));

  if (typeof node?.setSize === "function") {
    node.setSize([Number(width ?? 250), nextHeight]);
  } else if (Array.isArray(node?.size)) {
    node.size[1] = nextHeight;
  } else if (node?.size && typeof node.size === "object") {
    node.size[1] = nextHeight;
  }

  requestNodeRedraw(node);
  return nextHeight;
}
