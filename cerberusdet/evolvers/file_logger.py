import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml
from cerberusdet.utils.metrics import fitness, overall_fitness
from loguru import logger


class FileLogger:
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.yaml_file = self.save_dir / "hyp_evolved.yaml"  # save best result here
        self.evolve_results_file = str(self.save_dir / "evolve.json")  # save all evolutions here

    def append_mutation_to_file(
        self, hyp: Dict[str, Any], results_per_task: Dict[str, Tuple[float]], epoch: int, evolve_step: int
    ) -> None:

        print(
            f"\n{hyp}\nEvolved fitness: {[(k, fitness(v)) for k, v in results_per_task.items()]} "
            f"({overall_fitness(results_per_task)})\n"
        )

        # write new mutation to json file
        mutations_list = self.read_mutations()
        new_mutation = dict(results_per_task=results_per_task, epoch=epoch, hyps=hyp, evolve_step=evolve_step)
        mutations_list.append(new_mutation)
        self.write_mutations(mutations_list)

        # save best hyp to yaml file
        self.update_best_mutation()

    def update_best_mutation(self, fitness_func: Optional[Callable] = overall_fitness) -> None:
        mutations_list = self.read_mutations()

        # get best hyp by fitness func
        overall_fitness_for_gens = np.array([fitness_func(x["results_per_task"]) for x in mutations_list])
        best_ind = np.argmax(overall_fitness_for_gens)
        best_mutation = mutations_list[best_ind]

        # Save best hyps to yaml
        keys = ("P", "R", "mAP_0.5", "mAP_0.5:0.95", "box_loss", "cls_loss", "dfl_loss", "fitness")
        with open(self.yaml_file, "w") as f:
            out_str = "\n# " + " ".join(f"{x.strip():>10s}" for x in keys) + "\n"
            for task_name, results in best_mutation["results_per_task"].items():
                print(task_name, results)
                out_str += f"\n# {task_name}: "
                out_str += (
                    "%10.4g" * len(results) % tuple(results)
                )  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
                out_str += "%10.4g" % fitness(results)

            f.write(
                "# Hyperparameter Evolution Results\n"
                f"# Best generation: {best_ind+1}\n"
                f"# Generations: {len(mutations_list)}\n"
                f"# Metrics: {out_str}\n"
                f'# overall_fitness: {overall_fitness(best_mutation["results_per_task"]):.4f}\n\n'
            )

            yaml.safe_dump(best_mutation["hyps"], f, sort_keys=False)

    def is_last_mutation_best(self, fitness_func: Optional[Callable] = overall_fitness) -> bool:

        mutations_list = self.read_mutations()
        if len(mutations_list) == 0:
            return False

        # get best hyp by fitness func
        overall_fitness_for_gens = np.array([fitness_func(x["results_per_task"]) for x in mutations_list])
        best_ind = np.argmax(overall_fitness_for_gens)

        return best_ind == len(mutations_list) - 1

    def write_mutations(self, mutations_list: List[Dict[str, Any]]) -> None:
        with open(self.evolve_results_file, "w") as f:
            json.dump(mutations_list, f)

    def read_mutations(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.evolve_results_file):
            return []
        logger.info(f"Loading mutations_list from {self.evolve_results_file}")
        with open(self.evolve_results_file) as f:
            mutations_list = json.load(f)

        return mutations_list

    def read_top_5_mutations(self) -> List[Dict[str, Any]]:
        mutations_list = self.read_mutations()

        n = min(5, len(mutations_list))  # number of previous results to consider
        mutations_list = sorted(mutations_list, key=lambda x: overall_fitness(x["results_per_task"]), reverse=True)
        mutations_list = mutations_list[:n]  # top n mutations

        return mutations_list
