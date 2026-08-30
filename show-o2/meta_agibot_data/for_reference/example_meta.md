
Hi!

We’ve implemented a data reading handler for **agiworld-beta** using the **Lerobot** format (see [`datasets/domain_handler/lerobot_agibot.py`](datasets/domain_handler/lerobot_agibot.py)).
You can use this implementation as a **reference** to build your own handler or **directly utilize it** for dataset loading.

If you choose to use the existing handler directly, you need to prepare a **meta file** describing your dataset.
Below is a **demonstration format** (values are illustrative; please fill in according to your own data):

```jsonc
{
    // Name of the dataset
    "dataset_name": "AGIBOT",

    // A list of episode entries in the dataset
    "datalist": [
        {
            // Root path for this episode's data (local path, mount alias, or remote URI)
            "top_path": "path/to/your/data/task_xxx",

            // Index of the episode (starting from 0)
            "episode_index": 0,

            // Descriptive task information for this episode
            "tasks": [
                "Task description"
            ],

            // Total length (e.g., number of frames or steps)
            "length": ...,

            // List of action configurations (each defines a segment of the episode)
            "action_config": [
                {
                    "start_frame": ...,
                    "end_frame": ...,
                    "action_text": "Describe the action here.",
                    "skill": "Pick"
                },
                {
                    "start_frame": ...,
                    "end_frame": ...,
                    "action_text": "Describe another action.",
                    "skill": "Place"
                }
                // ...
            ]
        },
        // You can add more episodes below
        ...
    ]
}
```

### Explanation

* **`dataset_name`**: Identifier of your dataset (e.g., `"AGIBOT"`, `"MY_PROJECT"`).
* **`datalist`**: A list where each entry corresponds to one episode of collected data.

  * **`top_path`**: Base path to the episode’s raw data.
  * **`episode_index`**: Numerical index for episode tracking.
  * **`tasks`**: Human-readable task descriptions, possibly including environment context.
  * **`length`**: Total number of frames or timesteps in this episode.
  * **`action_config`**: (Optional) List describing individual action segments, each with start and end frames, textual description, and the associated skill.
