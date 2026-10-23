/*
 * The model, re-implemented in JavaScript so it can run inside the visitor's browser.
 *
 * It reads docs/model.json (written by src/export_web.py) and repeats the same arithmetic
 * scikit-learn does:  scaler -> (logistic regression + random forest + gradient boosting)
 * -> average of the three probabilities.  tests/test_web_parity.py checks that this file
 * gives the same answers as the Python model.
 *
 * It also contains a SHAP-style explainer (permutation method, same idea as the Python API).
 * The same code runs in Node (for the tests) and in the browser.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.Engine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const sigmoid = (v) => 1 / (1 + Math.exp(-v));

  // Walk one decision tree. scikit-learn compares in 32-bit precision, so we do too.
  function walk(tree, z) {
    let node = 0;
    while (tree.left[node] !== -1) {
      node = Math.fround(z[tree.feature[node]]) <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    return tree.value[node];
  }

  function scale(model, x) {
    const { mean, scale: sd } = model.scaler;
    const z = new Array(x.length);
    for (let j = 0; j < x.length; j++) z[j] = (x[j] - mean[j]) / sd[j];
    return z;
  }

  // Probability of malignancy for one patient whose measurements are already scaled (z).
  function probabilityFromScaled(model, z) {
    let lin = model.lr.intercept;
    for (let j = 0; j < z.length; j++) lin += model.lr.coef[j] * z[j];
    const pLr = sigmoid(lin);

    let sum = 0;
    const rf = model.rf.trees;
    for (let t = 0; t < rf.length; t++) sum += walk(rf[t], z);
    const pRf = sum / rf.length;

    let raw = 0;
    const gb = model.gb.trees;
    for (let t = 0; t < gb.length; t++) raw += walk(gb[t], z);
    const pGb = sigmoid(model.gb.start + model.gb.learning_rate * raw);

    return (pLr + pRf + pGb) / 3;
  }

  function predictProba(model, x) {
    return probabilityFromScaled(model, scale(model, x));
  }

  // A small, seedable random number generator, so the same patient always gets the same explanation.
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /*
   * Permutation SHAP. Idea: start from an "average patient" (every measurement taken from the
   * reference set) and swap in the real patient's measurements one at a time, in a random order.
   * The change in predicted probability at each swap is credited to that measurement.
   * Repeating over many random orders (each also run in reverse) gives each feature's average
   * contribution. Because the first and last states are fixed, the contributions always add up
   * exactly:  base_value + sum(contributions) = this patient's probability.
   */
  function explain(model, x, options) {
    const opts = Object.assign({ permutations: 40, seed: 42 }, options || {});
    const n = x.length;
    const zx = scale(model, x);
    const bg = model.background.map((row) => scale(model, row));
    const rows = bg.length;
    const rand = mulberry32(opts.seed);

    // Base value: the average prediction over the reference patients.
    let base = 0;
    for (let r = 0; r < rows; r++) base += probabilityFromScaled(model, bg[r]);
    base /= rows;
    const final = probabilityFromScaled(model, zx);

    const phi = new Array(n).fill(0);
    const order = Array.from({ length: n }, (_, i) => i);
    const scratch = bg.map((row) => row.slice());

    function sweep(seq) {
      // reset to the reference patients, then swap features in one by one
      for (let r = 0; r < rows; r++) for (let j = 0; j < n; j++) scratch[r][j] = bg[r][j];
      let before = base;
      for (let k = 0; k < n; k++) {
        const j = seq[k];
        let after;
        if (k === n - 1) {
          after = final; // every feature is now the patient's own
        } else {
          let total = 0;
          for (let r = 0; r < rows; r++) {
            scratch[r][j] = zx[j];
            total += probabilityFromScaled(model, scratch[r]);
          }
          after = total / rows;
        }
        phi[j] += after - before;
        before = after;
      }
    }

    for (let p = 0; p < opts.permutations; p++) {
      for (let i = n - 1; i > 0; i--) {
        const k = Math.floor(rand() * (i + 1));
        const tmp = order[i];
        order[i] = order[k];
        order[k] = tmp;
      }
      sweep(order);
      sweep(order.slice().reverse());
    }
    const shap = phi.map((v) => v / (2 * opts.permutations));
    return { base_value: base, shap_values: shap, probability: final };
  }

  // Everything the dashboard needs for one patient: diagnosis, warnings and explanation.
  function diagnose(model, features, options) {
    const names = model.feature_names;
    const x = names.map((n) => features[n]);
    const probability = predictProba(model, x);

    const warnings = [];
    names.forEach((name, j) => {
      const s = model.feature_stats[name];
      if (x[j] < s.min || x[j] > s.max) {
        warnings.push(
          `${name}=${+x[j].toPrecision(6)} is outside the range seen in the training data ` +
            `(${+s.min.toPrecision(6)} to ${+s.max.toPrecision(6)}); treat this prediction with caution.`
        );
      }
    });

    const result = { probability, prediction: probability >= model.threshold ? "Malignant" : "Benign", warnings };
    if (!options || options.explain !== false) {
      const e = explain(model, x, options);
      result.base_value = e.base_value;
      result.contributions = names
        .map((feature, j) => ({ feature, value: x[j], shap_value: e.shap_values[j] }))
        .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
    }
    return result;
  }

  return { predictProba, explain, diagnose, probabilityFromScaled, scale };
});
