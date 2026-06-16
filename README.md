<div align="center">
<h2>Position: The Systemic Lack of Agency in Visual Reasoning</h2>
Official repo for [ICML 2026] "Position: The Systemic Lack of Agency in Visual Reasoning"
</div>

<p align="center">
  <a href="#-abstract">Abstract</a> ·
  <a href="#-v-ird">Benchmark (V-IRD)</a> ·
  <a href="#-results">Results</a> ·
  <a href="#-citation">Citation</a>
</p>

## 🌞 Abstract

This paper argues that a systemic lack of Agency constrains the implicit reasoning capabilities of current Vision-Language Models (VLMs). Implicit reasoning refers to the ability to autonomously discover and utilize hidden visual evidence to bridge information gaps, rather than merely relying on explicitly specified targets. This capacity underlies human visual understanding and everyday reasoning. To address this gap, we introduce the Visual Implicit Reasoning Diagnosing Benchmark (V-IRD), which compels models to derive answers strictly through autonomous visual analysis. Our results show that, despite strong retrieval abilities, prominent VLMs struggle to utilize reference objects and to attend to visual evidence that requires self-directed inquiry. Simply put, strong semantic recognition does not equate to active visual exploration, revealing a critical gap in current VLMs.

<div align="center">
  <img src="static/images/f1.png" width="80%">
</div>

<div align="center">

**Figure 1.  Comparison of active vs. passive capacity in visual reasoning**

</div>

---

## 📖 V-IRD

This benchmark is divided into four core categories:

* **Spatial Geometry:** Focuses on precise metrology tasks such as Length, Distance, Volume, and Area.
* **Contextual Inference:** Challenges the model to deduce abstract information like Environment and Remark.
* **Physical Properties:** Covers thermal states (Temperature) and Weight.
* **Physical Logic:** Involves abstract reasoning like Electricity and Kinematics.

<div align="center">
  <img src="static/images/task_distribution.png" width="80%">
</div>

<div align="center">

**Figure 2. Statistics of categories and tasks in V-IRD.**

</div>

---

## 🍭 Results

<div align="center">
  <img src="static/images/main_results_table.png" width="80%">
</div>

<div align="center">

**Figure 3. Main results on V-IRD**

</div>

---

## ⭐ Citation

If you find this research or the V-IRD benchmark helpful, please cite our paper as follows:

```bibtex
