/* -----------------------------------------------------------------
   Helper functions
   ----------------------------------------------------------------- */
function log(msg) {
  const l = document.getElementById('log');
  l.textContent += msg + '\n';
  l.scrollTop = l.scrollHeight;
}

/* -----------------------------------------------------------------
   1️⃣  Load the node schema (INPUT_TYPES) from the backend
   ----------------------------------------------------------------- */
async function fetchNodeSchema() {
  // Every node module can be queried via /custom-node/:module_name/schema
  // The module name is the python file name without extension.
  const resp = await fetch('/custom-node/TBG_MAGNIFIC_MAGNIFIER/schema');
  if (!resp.ok) throw new Error('Failed to get schema');
  return await resp.json();   // returns the same dict as INPUT_TYPES()
}

/* -----------------------------------------------------------------
   2️⃣  Build a simple HTML form from that schema
   ----------------------------------------------------------------- */
function buildForm(schema) {
  const form = document.getElementById('node-form');
  form.innerHTML = ''; // clear any previous UI

  // Helper to create a label+control pair
  const addField = (name, spec, container = form) => {
    const div = document.createElement('div');
    const label = document.createElement('label');
    label.htmlFor = name;
    label.textContent = spec.label || name;
    div.appendChild(label);

    const type = spec[0]; // e.g. "INT", "FLOAT", "STRING", "IMAGE", …
    const opts = spec[1] || {};

    // -------------------------------------------------------------
    // Very small subset of UI types – enough for this node.
    // -------------------------------------------------------------
    let input;
    if (type === "INT" || type === "FLOAT") {
      input = document.createElement('input');
      input.type = 'number';
      if (type === "FLOAT") input.step = opts.step || 0.01;
      if (opts.min !== undefined) input.min = opts.min;
      if (opts.max !== undefined) input.max = opts.max;
      if (opts.default !== undefined) input.value = opts.default;
    } else if (type === "STRING") {
      if (opts.multiline) {
        input = document.createElement('textarea');
        input.rows = 3;
        input.value = opts.default || "";
      } else {
        input = document.createElement('input');
        input.type = 'text';
        input.value = opts.default || "";
      }
    } else if (type === "BOOLEAN") {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = !!opts.default;
    } else if (type === "IMAGE") {
      input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
    } else if (Array.isArray(type)) {        // enum: ["foo","bar"]
      input = document.createElement('select');
      type.forEach(val => {
        const opt = document.createElement('option');
        opt.value = val;
        opt.textContent = val;
        if (opts.default && opts.default === val) opt.selected = true;
        input.appendChild(opt);
      });
    } else {
      // fallback – treat as text
      input = document.createElement('input');
      input.type = 'text';
    }

    input.id = name;
    input.name = name;
    div.appendChild(input);
    container.appendChild(div);
  };

  // The schema is split in “required”, “optional” and “hidden”.
  // Hidden fields are auto‑filled by the UI code later, so we skip them here.
  for (const group of ["required", "optional"]) {
    const params = schema[group] || {};
    for (const [key, spec] of Object.entries(params)) {
      addField(key, spec);
    }
  }
}

/* -----------------------------------------------------------------
   3️⃣  Convert form values to the JSON format expected by ComfyUI
   ----------------------------------------------------------------- */
function getFormData() {
  const data = {};
  const formElements = document.getElementById('node-form').elements;
  for (const el of formElements) {
    if (!el.name) continue;
    const type = el.type;
    if (type === 'checkbox') {
      data[el.name] = el.checked;
    } else if (type === 'file') {
      // For images we will read the file as a base64 data‑uri later.
      data[el.name] = el.files[0] || null;
    } else if (type === 'number') {
      data[el.name] = el.value === '' ? null : Number(el.value);
    } else {
      data[el.name] = el.value;
    }
  }
  return data;
}

/* -----------------------------------------------------------------
   4️⃣  Helper – upload a single image file to the server & obtain its
       “image” object (the same structure ComfyUI uses internally)
   ----------------------------------------------------------------- */
async function uploadImage(file) {
  const form = new FormData();
  form.append('image', file);
  const resp = await fetch('/upload/image', { method: 'POST', body: form });
  if (!resp.ok) throw new Error('image upload failed');
  const json = await resp.json();
  // The response format is: { "filename": "...", "type": "output", "subfolder": "", "save_to_file": false }
  // and the same object is what we need to place into the workflow.
  return json;
}

/* -----------------------------------------------------------------
   5️⃣  Build the *single‑node* workflow JSON that ComfyUI expects.
   ----------------------------------------------------------------- */
async function buildWorkflow(formData) {
  // The node class name is the one that ComfyUI registers (the static
  // CATEGORY & NAME are irrelevant here). It is the python class name.
  const nodeClass = "TBG_magnific_ETUR";

  // 5️⃣.1 – Resolve images (if any) to the /upload format.
  const inputs = { ...formData };
  for (const [k, v] of Object.entries(inputs)) {
    if (v instanceof File) {
      inputs[k] = await uploadImage(v);
    }
  }

  // 5️⃣.2 – Hidden fields that ComfyUI expects for every node.
  const hidden = {
    id: "unique_node_id_" + Date.now(),
    // These three are mandatory for a *stand‑alone* prompt.
    // They are usually added by the UI, we just use temporary placeholders.
    extra_pnginfo: {},
    prompt: {}
  };

  // 5️⃣.3 – The final payload.
  const workflow = {
    // The top‑level prompt dictionary – a map from node‑ID to its definition.
    // Here we only have ONE node.
    "prompt": {
      [hidden.id]: {
        "class_type": nodeClass,
        "inputs": inputs
      }
    },
    // Tell the executor which node(s) we want the result from.
    // Empty list means “run everything and give you the whole graph”.
    "output": [hidden.id],
    // These two fields are required for the UI but can be empty objects.
    "extra_pnginfo": hidden.extra_pnginfo,
    "workflow": {}   // not used by the backend for a single‑node run.
  };
  return workflow;
}

/* -----------------------------------------------------------------
   6️⃣  Submit the workflow to /prompt and listen for the result
   ----------------------------------------------------------------- */
async function runWorkflow() {
  document.getElementById('results').innerHTML = '';
  log('Collecting form data …');
  const formData = getFormData();

  log('Building workflow …');
  const workflow = await buildWorkflow(formData);

  // POST the workflow to the engine
  log('Sending prompt …');
  const resp = await fetch('/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(workflow)
  });
  if (!resp.ok) throw new Error('Prompt submission failed');

  const respJson = await resp.json();
  const promptId = respJson.prompt_id;
  log(`Prompt submitted, id = ${promptId}`);

  // -------------------------------------------------------------
  // 6️⃣.1  Listen for the completion event on the websocket
  // -------------------------------------------------------------
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    log('WebSocket opened …');
    // Subscribe to status updates for *all* prompts (filter on id later)
    ws.send(JSON.stringify({ "type": "subscribe", "data": {} }));
  };
  ws.onmessage = event => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'status' && msg.prompt_id === promptId) {
      if (msg.status === 'executed') {
        // The prompt finished – now retrieve the results.
        fetch(`/history/${promptId}`)
          .then(r => r.json())
          .then(history => {
            const nodeId = Object.keys(history.outputs)[0];
            const output = history.outputs[nodeId];
            // `output` is an object where the keys correspond to the return
            // names defined in RETURN_NAMES (STEP1 Refined Image …)
            displayResults(output);
          })
          .catch(e => log('Failed to fetch history: ' + e));
        ws.close();
      } else if (msg.status === 'failed') {
        log('Prompt failed – see console for details.');
        ws.close();
      }
    }
  };
  ws.onerror = err => log('WebSocket error: ' + err);
}

/* -----------------------------------------------------------------
   7️⃣  Render the five output images that the node returns
   ----------------------------------------------------------------- */
function displayResults(output) {
  const container = document.getElementById('results');
  container.innerHTML = '';

  // The node returns a tuple of five images – they are stored under
  // keys that match the RETURN_NAMES list.
  const names = [
    "STEP1 Refined Image",
    "STEP2 Refined Image",
    "STEP3 Refined Image",
    "STEP4 Refined Image",
    "Refinement or Final Image"
  ];

  names.forEach((name, idx) => {
    const imgData = output[idx];
    if (!imgData) return;   // could be `null` if the node left a placeholder

    const imgUrl = `/view?filename=${encodeURIComponent(imgData.filename)}&type=${encodeURIComponent(imgData.type)}&subfolder=${encodeURIComponent(imgData.subfolder)}`;
    const div = document.createElement('div');
    const caption = document.createElement('p');
    caption.textContent = name;
    const img = document.createElement('img');
    img.src = imgUrl;
    img.alt = name;
    div.appendChild(caption);
    div.appendChild(img);
    container.appendChild(div);
  });
}

/* -----------------------------------------------------------------
   8️⃣  Initialise UI on page load
   ----------------------------------------------------------------- */
document.addEventListener('DOMContentLoaded', async () => {
  try {
    const schema = await fetchNodeSchema();
    buildForm(schema);
    document.getElementById('run-btn').addEventListener('click', async () => {
      try {
        await runWorkflow();
      } catch (e) {
        log('❌ ' + e);
        console.error(e);
      }
    });
    document.getElementById('reset-btn').addEventListener('click', () => {
      document.getElementById('results').innerHTML = '';
      document.getElementById('log').textContent = '';
    });
    log('UI ready – fill the fields and press **Run**.');
  } catch (e) {
    log('Failed to load schema: ' + e);
    console.error(e);
  }
});
