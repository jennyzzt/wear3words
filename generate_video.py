import replicate


model = replicate.models.get("nateraw/stable-diffusion-videos")
print('got model')
version = model.versions.get("2d87f0f8bc282042002f8d24458bbf588eee5e8d8fffb6fbb10ed48d1dac409e")
print('got version')


def words_to_video(words):
    words_list = ' | '.join(words)
    inputs = {
        'prompts': words_list,
        'scheduler': "klms",
        'num_inference_steps': 1,
        'guidance_scale': 7.5,
        'num_steps': 3,
        'fps': 5,
    }
    output = version.predict(**inputs)
    return output


if __name__ == "__main__":
    # https://replicate.com/nateraw/stable-diffusion-videos/versions/2d87f0f8bc282042002f8d24458bbf588eee5e8d8fffb6fbb10ed48d1dac409e#input
    inputs = {
        # Input prompts, separate each prompt with '|'.
        'prompts': "a cat | a dog | a horse",

        # Random seed, separated with '|' to use different seeds for each of
        # the prompt provided above. Leave blank to randomize the seed.
        # 'seeds': ...,

        # Choose the scheduler
        'scheduler': "klms",

        # Number of denoising steps for each image generated from the prompt
        # Range: 1 to 500
        'num_inference_steps': 5,

        # Scale for classifier-free guidance
        # Range: 1 to 20
        'guidance_scale': 7.5,

        # Steps for generating the interpolation video. Recommended to set to
        # 3 or 5 for testing, then up it to 60-200 for better results.
        'num_steps': 3,

        # Frame rate for the video.
        # Range: 5 to 60
        'fps': 5,
    }

    # https://replicate.com/nateraw/stable-diffusion-videos/versions/2d87f0f8bc282042002f8d24458bbf588eee5e8d8fffb6fbb10ed48d1dac409e#output-schema
    output = version.predict(**inputs)
    print(output)
