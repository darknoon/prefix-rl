from datasets import load_dataset

# TODO!


def filter_svg_stack_dataset(dataset):
    """
    from the paper:
    We begin by filtering the SVG-Stack dataset to select 20k high-entropy samples that are rich in visual detail and SVG complexity (each with over 500 tokens).

    Curating SVG Stack To train our model, we require large-scale data to capture the fundamental
    structures of SVG. For this, we leverage the SVG-Stack dataset [Rodriguez et al., 2025b], which
    consists of SVG files scraped from GitHub. We rasterize these files to obtain paired image-SVG
    examples.
    The original SVG-Stack contains 2.1M samples. We preprocess this data by rounding decimals to two
    significant figures, removing XML headers, and filtering out samples with excessively long URLs or
    embedded base64 images, which could lead the model to memorize irrelevant content. After cleaning,
    we retain 1.7M high-quality examples, which we use to train the model in the SVG-SFT stage, where
    it learns to generate SVGs from images as faithfully as possible. The SVGs are rendered using
    CairoSVG [Kozea, 2023], and image augmentations follow the protocol introduced by LLaVA [Liu
    et al., 2023]."""
    pass


def main():
    dataset = load_dataset("MrOvkill/svg-stack-labeled")
    filtered_dataset = filter_svg_stack_dataset(dataset)
    print(filtered_dataset)


if __name__ == "__main__":
    main()
