# trunkplot

A simple visualisation style for model architectures (implemented for HF models, tested for LLMs{text,audio,multimodal}).

- general
- component-wise color coding
- relative parameter size comparision within model at log scale
- represents model layers / operation in order visually


<img src="https://github.com/user-attachments/assets/6eed874d-940b-4278-a8a1-f2e2b443c9c8" width="400"/>


## method

1. **Clean Layer Names**  
   Example: Convert `encoder.layer.0.attention.self.query.weight` to `encoder.layer.attention.self.query.weight`.

2. **Compute Parameter Counts**  
   Example: `query.weight` may have 768 x 768 = 589,824 parameters.

3. **Normalize Sizes Logarithmically**  
   Scale parameter counts using logarithms to fit visualization constraints.
   
   Example: If the smallest layer has 100K params and the largest has 10M, map their cube sizes proportionally.

4. **Assign Unique Colors**  
   Use a colormap to distinguish layer types.
   
   Example: Attention layers in blue, feed-forward layers in red, embedding layers in green (not hardcoding it right now)

## Application

### Initial Rough Versions

#### gpt2-small

<img src="https://github.com/user-attachments/assets/c884debd-e0f2-4e72-a747-5ecf60da8ae8" width="440" height="540">
<img src="https://github.com/user-attachments/assets/c3070b51-c1c0-48a8-b86f-afe5bd49bd24" width="440" height="540">

### Refined Version After Few Iterations

<img src="https://github.com/user-attachments/assets/4f0cbd97-1dfa-4f55-b2b2-756e3b820235" width="440" height="540">

### Method Applied to Popular Models

<img src="https://github.com/user-attachments/assets/335a9cc3-cadd-4aaa-9525-26ae4078227b" width="440" height="540">
<img src="https://github.com/user-attachments/assets/fbdd74b8-100d-416c-b49a-9c366c0d3a12" width="440" height="540">
<img src="https://github.com/user-attachments/assets/d5dcf135-3622-4499-b6da-1cd58dd21774" width="440" height="540">
<img src="https://github.com/user-attachments/assets/10587a15-e140-4eb9-abbc-e14c76d5685f" width="440" height="540">
<img src="https://github.com/user-attachments/assets/e1d7f1cc-8470-4675-8d0d-88ddeaefc6db" width="440" height="540">
<img src="https://github.com/user-attachments/assets/3e45518f-9d6e-42a1-af2b-941c5436ed9e" width="440" height="540">


<br>
Note: more in diagrams/ folder of this repo

<br>
<br>

[closeup](https://x.com/attentionmech/status/1907714094817284578)

<br>

[More on this thread](https://x.com/attentionmech/status/1907538095899042173)


#### how to run

```
uv run https://raw.githubusercontent.com/attentionmech/trunkplot/refs/heads/main/main.py <model_name>
```

examples:
```
uv run https://raw.githubusercontent.com/attentionmech/trunkplot/refs/heads/main/main.py gpt2
```

if you checked out
```
uv run main.py gpt2
```

## Citation

```
@article{attentionmech2025trunkplot,
  title={trunkplot: a simple visualisation style for model architectures},
  author={attentionmech},
  year={2025}
}
```

