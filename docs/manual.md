# Manual

```powershell
.\recipe.ps1 run
```

Run it from the root of the repository: the image path `data/image.png` is relative. The script takes no options. To classify another image or other labels, change `image_path`, `candidate_labels` or `model_name` in `main()`.

A run logs to the terminal, at level INFO, which model it loads on which device and which image it classifies, then prints two lines:

```text
Predicted index: 0
Predicted label: a car
```

The index counts from 0 in `candidate_labels`. Each label is scored as the prompt `This is a photo of <label>.`, and the label with the highest score wins; the scores themselves are not printed.

When anything fails, from a missing image to weights that cannot be fetched, the script logs `An error occurred:` with the reason and prints no prediction. It still exits 0.
