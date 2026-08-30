# How to Ground Manipulation Foundation Models? An Empirical Study Under A Unified Vision-Language-World-Action Framework

## Visualized Grounding + Future Generation

- showvla-future_act_weighted_grounding_jaka0430

20260504 weighted LIBERO + JAKA + JAKA-moveto, grounding LIBERO-Spatial, LIBERO-90, JAKA-clutter
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 94.4    | 99.6   | 98.4  | 94.2  | 96.7  |             | 77.0    | 68.0   | 70.6  | 66.3  | 70.5  |
| 19000 | 93.0    | 98.2   | 99.0  | 94.2  | 96.1  |             |     |    |   |   |   |
| 20000 | 94.6    | 99.8   | 95.0  | 95.4  | 96.2  |             |     |    |   |   |   |


- showvla-future_act_weighted_grounding_jaka0601_sb

20260611 weighted LIBERO + JAKA + JAKA-moveto (small batch)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 94.8    | 100.   | 97.2  | 93.2  | 96.3  |             |     |    |   |   |   |
| 19000 | 93.0    | 100.   | 98.2  | 95.4  | 96.7  |             |     |    |   |   |   |
| 20000 | 95.8    | 98.6   | 97.4  | 96.6  | 97.1  |             | 74.2    | 65.0   | 72.3  | 65.7  | 69.3  |

- showvla-future_act_weighted_grounding_jaka_calvin

20260611 weighted LIBERO + JAKA + JAKA-moveto + CALVIN
(T//2 per traj., jaka domain id: 29->15)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 92.0    | 98.4   | 97.8  | 97.2  | 96.4  |             |     |    |   |   |   |
| 19000 | 92.8    | 98.8   | 98.0  | 96.2  | 96.5  |             |     |    |   |   |   |
| 20000 | 94.2    | 100.   | 98.2  | 97.6  | 97.5  |             | 70.5    | 66.9   | 65.5  | 61.4  | 66.1  |

- showvla-future_act_weighted_grounding_jaka_calvin0615

20260616 weighted LIBERO + JAKA + JAKA-moveto + CALVIN
(T//2 per traj., jaka domain id: 29->15, mid batch)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 94.6    | 99.0   | 98.8  | 95.6  | 97.0  |             |     |    |   |   |   |
| 19000 | 92.4    | 99.8   | 97.2  | 92.6  | 95.5  |             |     |    |   |   |   |
| 20000 | 95.0    | 99.6   | 98.6  | 95.6  | 97.2  |             |     |    |   |   |   |

- showvla-future_act_weighted_grounding_jaka_calvin0616

20260618 weighted LIBERO + JAKA + JAKA-moveto + CALVIN
(T//2 per traj., jaka domain id: 29->15, mid batch, num_actions: 24->20)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 94.0    | 92.8   | 98.0  | 94.4  | 94.8  |             |     |    |   |   |   |
| 19000 | 91.0    | 98.0   | 99.2  | 90.8  | 94.8  |             |     |    |   |   |   |
| 20000 | 96.6    | 98.4   | 98.6  | 94.6  | 97.1  |             |     |    |   |   |   |

- showvla-future_act_weighted_grounding_jaka_calvin_lvis0619

20260621 weighted LIBERO + JAKA + JAKA-moveto + CALVIN
(T//2 per traj., jaka domain id: 29->15, mid batch, lvis w/o combine)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 96.2    | 97.6   | 98.8  | 95.8  | 97.1  |             | 77.6    | 51.6   | 72.1  | 62.5  | 66.2  |
| 19000 | 96.8    | 99.0   | 98.8  | 94.8  | 97.4  |             | 78.1    | 56.7   | 72.3  | 62.3  | 67.4  |
| 20000 | 96.2    | 99.6   | 98.6  | 96.8  | 97.8  |             | 76.9    | 56.0   | 71.9  | 55.2  | 65.0  |

- showvla-future_act_weighted_grounding_jaka_lvis0621

20260622 weighted LIBERO + JAKA + JAKA-moveto
(T//2 per traj., jaka domain id: 29->15, mid batch, lvis w/o combine)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 95.6    | 96.4   | 98.6  | 96.2  | 96.7  |             | 76.5    | 62.9   | 71.5  | 66.7  | 69.4  |
| 19000 | 95.8    | 99.6   | 99.2  | 94.4  | 97.3  |             | 76.3    | 69.0   | 69.5  | 69.5  | 71.1  |
| 20000 | 94.2    | 100.0  | 98.6  | 95.6  | 97.1  |             | 76.7    | 69.2   | 70.0  | 71.6  | 71.9  |

- showvla-future_act_weighted_grounding_jaka_lvis0624

20260626 weighted LIBERO + JAKA + JAKA-moveto
(T//2 per traj., jaka domain id: 29->15, mid batch, both w/o combine)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 90.8    | 99.8   | 97.8  | 94.2  | 95.6  |             | 74.7    | 73.6   | 72.4  | 65.5  | 71.5  |
| 19000 | 91.0    | 100.0  | 97.2  | 91.0  | 94.8  |             | 76.4    | 74.8   | 72.5  | 65.8  | 72.4  |
| 20000 | 92.0    | 99.2   | 97.0  | 93.6  | 95.4  |             | 78.1    | 74.7   | 72.1  | 64.6  | 72.4  |

- showvla-future_act_weighted_grounding_jaka_lvis0626

20260628 weighted LIBERO + JAKA + JAKA-moveto
(T//2 per traj., jaka domain id: 29->15, mid batch, replicate results of lvis0621)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 91.6    | 99.4   | 97.8  | 95.4  | 96.0  |             | 76.3    | 79.0   | 72.3  | 64.4  | 73.0  |
| 19000 | 93.2    | 99.8   | 96.4  | 92.0  | 95.3  |             | 77.6    | 79.7   | 71.5  | 70.6  | 74.9  |
| 20000 | 91.2    | 99.4   | 97.4  | 93.4  | 95.3  |             | 77.8    | 79.0   | 72.7  | 69.4  | 74.8  |

- showvla-future_act_weighted_grounding_jaka_lvis0629

20260630 weighted LIBERO + JAKA + JAKA-moveto
(T//2 per traj., jaka domain id: 29->15, mid batch, replicate results of lvis0621)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 94.2    | 99.4   | 98.2  | 95.0  | 96.7  |             | 72.8    | 69.9   | 70.7  | 65.3  | 69.7  |
| 19000 | 90.4    | 99.0   | 98.6  | 91.8  | 94.9  |             |     |    |   |   |   |
| 20000 | 96.2    | 98.6   | 97.6  | 92.2  | 96.1  |             | 73.0    | 71.4   | 70.7  | 67.3  | 70.6  |

- showvla-future_act_weighted_grounding_jaka_calvin_lvis

20260611 weighted LIBERO + JAKA + JAKA-moveto + CALVIN (T//2 per traj., jaka domain id: 29->15)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 90.8    | 99.6   | 96.8  | 98.0  | 96.3  |             |     |    |   |   |   |
| 19000 | 93.6    | 98.0   | 98.0  | 94.4  | 96.0  |             |     |    |   |   |   |
| 20000 | 94.2    | 99.6   | 97.6  | 89.0  | 95.1  |             |     |    |   |   |   |

- showvla-future_act_weighted_grounding_jaka_calvin_bridge_lvis0613

20260614 weighted LIBERO + JAKA + JAKA-moveto + CALVIN + Bridge (T//2 per traj., jaka domain id: 29->15)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 91.2    | 99.4   | 98.0  | 95.6  | 96.1  |             |     |    |   |   |   |
| 19000 | 92.8    | 99.4   | 97.2  | 95.2  | 96.2  |             |     |    |   |   |   |
| 20000 | 93.6    | 99.2   | 97.8  | 92.6  | 95.8  |             |     |    |   |   |   |

- showvla-future_act_weighted_grounding_jaka-calvin-bridge-robomind_lvis

20260702 weighted LIBERO + JAKA + JAKA-moveto + (CALVIN + Bridge (stage1 only))
(T//2 per traj., jaka domain id: 29->15, mid batch, lvis w/o combine)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 92.6    | 99.8   | 98.0  | 97.4  | 97.0  |             | 75.3    | 65.5   | 71.7  | 63.3  | 69.0  |
| 19000 | 94.8    | 100.0  | 97.6  | 94.6  | 96.8  |             | 77.5    | 68.7   | 68.9  | 63.6  | 69.7  |
| 20000 | 91.0    | 99.6   | 97.8  | 95.0  | 95.9  |             | 73.7    | 64.9   | 69.7  | 69.1  | 69.4  |

- showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid_lvis

20260721 weighted LIBERO + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid (stage1+stage2_warmup))
(T//2 per traj., jaka domain id: 29->15, mid batch, lvis w/o combine)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 23000 | 90.0    | 93.6   | 98.2  | 95.6  | 94.4  |             |     |    |   |   |   |
| 24000 | 93.0    | 97.6   | 98.8  | 95.6  | 96.3  |             |     |    |   |   |   |
| 25000 | 92.8    | 99.0   | 96.2  | 95.0  | 95.8  |             |     |    |   |   |   |

- showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis

20260814 weighted LIBERO + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid + AGIBOT (stage1+stage2_warmup))
(T//2 per traj., jaka domain id: 29, mid batch, lvis w/o combine)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 23000 | 93.8    | 99.8   | 98.6  | 93.0  | 96.3  |             | 74.0    | 66.5   | 69.5  | 58.6  | 67.2  |
| 24000 | 91.4    | 99.8   | 97.8  | 96.2  | 96.3  |             | 73.3    | 70.6   | 68.8  | 66.7  | 69.9  |
| 25000 | 96.6    | 98.6   | 97.6  | 94.4  | 96.8  |             | 75.2    | 67.0   | 69.8  | 64.9  | 69.2  |

- showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0815

20260817 weighted LIBERO + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid + AGIBOT (stage1+stage2_warmup))
(T//2 per traj., jaka domain id: 29, mid batch, lvis w/o combine, add context for action head)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 23000 | 97.6    | 98.4   | 97.4  | 96.2  | 97.4  |             | 76.8    | 69.2   | 69.0  | 69.4  | 71.1  |
| 24000 | 97.8    | 98.8   | 99.4  | 94.4  | 97.6  |             | 77.0    | 70.2   | 68.7  | 68.0  | 71.0  |
| 25000 | 96.4    | 99.8   | 98.2  | 93.0  | 96.9  |             | 77.1    | 69.9   | 69.0  | 68.0  | 71.0  |


- showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0816

20260819 mujoco332
weighted LIBERO + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid + AGIBOT (stage1 only))
(T//2 per traj., jaka domain id: 29, mid batch, lvis w/o combine, add context for action head)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 17000 | 99.4    | 99.0   | 98.0  | 95.4  | 98.0  |             |     |    |   |   |   |
| 18000 | 97.2    | 99.4   | 95.6  | 92.6  | 96.2  |             |     |    |   |   |   |
| 19000 | 97.4    | 99.8   | 97.8  | 95.2  | 97.6  |             |     |    |   |   |   |
| 20000 | 99.0    | 99.4   | 98.4  | 93.2  | 97.5  |             |     |    |   |   |   |

20260820 mujoco337
weighted LIBERO + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid + AGIBOT (stage1 only))
(T//2 per traj., jaka domain id: 29, mid batch, lvis w/o combine, add context for action head)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 17000 | 98.6    | 99.4   | 98.0  | 96.2  | 98.1  |             | 77.3    | 54.1   | 66.7  | 60.4  | 64.6  |
| 18000 | 97.8    | 99.8   | 96.2  | 93.2  | 96.8  |             | 76.3    | 56.7   | 64.7  | 59.7  | 64.4  |
| 19000 | 98.4    | 99.2   | 98.4  | 96.8  | 98.2  |             | 76.3    | 61.7   | 65.5  | 59.2  | 65.7  |
| 20000 | 98.6    | 99.8   | 98.4  | 92.6  | 97.4  |             | 77.5    | 61.3   | 67.5  | 59.4  | 66.4  |


- showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0820

20260821 mujoco332
weighted LIBERO (change to mujoco332 in stage2) + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid + AGIBOT (stage1 only))
(T//2 per traj., jaka domain id: 29, mid batch, lvis w/o combine, add context for action head)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 17000 | 98.6    | 98.0   | 98.2  | 94.0  | 97.2  |             | 72.1    | 61.1   |       |       |       |
| 18000 | 99.8    | 99.2   | 99.4  | 94.2  | 98.2  |             | 71.7    | 61.8   |       |       |       |
| 19000 | 99.2    | 99.0   | 97.4  | 93.8  | 97.4  |             | 73.7    | 74.1   |       |       |       |
| 20000 | 98.8    | 99.2   | 98.2  | 91.6  | 97.0  |             | 76.3    | 73.0   |       |       |       |


- showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0820_

20260822 mujoco332 (replicate of showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0820)
weighted LIBERO (change to mujoco332 in stage2) + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid + AGIBOT (stage1 only))
(T//2 per traj., jaka domain id: 29, mid batch, lvis w/o combine, add context for action head)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 17000 | 97.6    | 99.0   | 95.2  | 96.4  | 97.1  |             | 71.1    | 53.1   | 58.4  | 59.3  | 60.5  |
| 18000 | 98.6    | 99.6   | 98.2  | 90.0  | 96.6  |             | 72.6    | 51.7   | 62.9  | 60.1  | 61.8  |
| 19000 | 98.4    | 99.0   | 99.0  | 94.2  | 97.7  |             | 72.6    | 57.2   | 63.9  | 58.7  | 63.1  |
| 20000 | 98.2    | 100.0  | 98.4  | 95.8  | 98.1  |             | 73.1    | 55.9   | 62.3  | 64.1  | 63.9  |


- showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0823

20260825 mujoco332
weighted LIBERO (fully change to mujoco332) + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid + AGIBOT (stage1 only))
(T//2 per traj., jaka domain id: 29, mid batch, lvis w/o combine, add context for action head)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 17000 | 99.8    | 99.4   | 97.6  | 92.0  | 97.2  |             | 77.9    | 71.1   | 69.7  | 58.8  | 69.4  |
| 18000 | 97.4    | 98.6   | 97.8  | 96.6  | 97.6  |             | 77.6    | 69.9   | 69.8  | 63.9  | 70.3  |
| 19000 | 98.4    | 98.2   | 95.6  | 95.2  | 96.9  |             | 77.3    | 66.9   | 69.4  | 64.6  | 69.6  |
| 20000 | 96.8    | 99.8   | 95.4  | 96.6  | 97.2  |             | 76.1    | 72.4   | 67.9  | 64.9  | 70.3  |


- showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0825

20260829 mujoco332
weighted LIBERO (fully change to mujoco332) + JAKA + JAKA-moveto + (CALVIN + Bridge + robomind + droid + AGIBOT (stage1 only))
(T//2 per traj., jaka domain id: 29, mid batch, lvis w/o combine, add context for action head)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 17000 | 99.0    | 99.2   | 97.2  | 96.8  | 98.1  |             |     |    |   |   |   |
| 18000 | 97.6    | 98.2   | 98.6  | 91.6  | 96.5  |             |     |    |   |   |   |
| 19000 | 97.4    | 99.4   | 98.0  | 94.6  | 97.4  |             |     |    |   |   |   |
| 20000 | 97.0    | 99.2   | 96.8  | 94.4  | 96.9  |             |     |    |   |   |   |


## VQA Grounding + Future Generation

- showvla-future_act_weighted_VQAgrounding_jaka0503

20260508 weighted LIBERO + JAKA + JAKA-moveto, VQA grounding LIBERO-Spatial, LIBERO-90, JAKA-clutter (small batch)
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 95.6    | 98.6   | 98.0  | 93.0  | 96.3  |             |     |    |   |   |   |
| 19000 | 96.0    | 99.2   | 98.8  | 94.8  | 97.2  |             | 75.8    | 68.7   | 68.0  | 68.8  | 70.3  |
| 20000 | 96.8    | 99.8   | 99.2  | 92.0  | 97.0  |             |     |    |   |   |   |

- showvla-future_act_weighted_VQAgrounding_jaka0506

20260514 weighted LIBERO + JAKA + JAKA-moveto, VQA grounding LIBERO-Spatial, LIBERO-90, JAKA-clutter
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 89.2    | 98.6   | 98.4  | 92.4  | 94.7  |             | 68.0    | 56.3   | 66.6  | 66.9  | 64.5  |
| 19000 | 90.6    | 99.8   | 98.8  | 95.4  | 96.2  |             | 69.4    | 57.7   | 68.1  | 65.7  | 65.2  |
| 20000 | 94.0    | 99.6   | 98.4  | 92.4  | 96.1  |             | 68.9    | 58.4   | 68.2  | 60.7  | 64.1  |


## Visualized Grounding + VQA Grounding + Future Generation

- showvla-future_act_weighted_COMBgrounding_jaka

20260524 weighted LIBERO + JAKA + JAKA-moveto, Combined grounding LIBERO-Spatial, LIBERO-90, JAKA-clutter
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 96.8    | 95.2   | 95.0  | 85.2  | 93.1  |             |     |    |   |   |   |
| 19000 | 92.4    | 95.8   | 97.2  | 95.2  | 95.2  |             |     |    |   |   |   |
| 20000 | 89.2    | 98.8   | 97.6  | 75.0  | 90.2  |             |     |    |   |   |   |

# Future Generation

- showvla-future_act_weighted_jaka0522

20260526 weighted LIBERO + JAKA + JAKA-moveto
|       | Spatial | Object | Goal  | Long  | AVG.  | LIBERO-plus | Spatial | Object | Goal  | Long  | AVG.  |
| :---: | :---:   | :---:  | :---: | :---: | :---: | :---:       | :---:   | :---:  | :---: | :---: | :---: |
| 18000 | 89.4    | 98.2   | 95.6  | 82.0  | 91.3  |             |     |    |   |   |   |
| 19000 | 86.4    | 97.2   | 93.6  | 79.0  | 89.0  |             |     |    |   |   |   |
| 20000 | 91.6    | 88.8   | 92.2  | 75.4  | 87.0  |             |     |    |   |   |   |


# Data

Converted metainfo JSON files are consumed by training configs under `configs/`. Image-level grounding metas live in `meta_grounding_data/`; robot trajectory metas live in `meta_*_data/` and are copied into a `meta_join_libero_jaka/` split used by the run.

Replace `PATH/TO/...` below with local paths. Example shell scripts under each `meta_*_data/` directory hardcode machine-specific paths and should be edited first.

## Grounding Data

### COCO

Download COCO 2017 (`annotations`, `train2017`, `val2017`).

```bash
cd ShowVLA/show-o2/meta_grounding_data/coco

python coco_data_convert.py --data_root PATH/TO/COCO --split_name train
python coco_data_convert.py --data_root PATH/TO/COCO --split_name val
```

Example: `meta_grounding_data/coco/coco_data_convert.sh`. Outputs `coco_train2017_meta.json` / `coco_val2017_meta.json`.

### ADE20K

Download ADEChallengeData2016. Split names are `training` / `validation` (not `train` / `val`).

```bash
cd ShowVLA/show-o2/meta_grounding_data/ade20k

python ade20k_data_convert.py --data_root PATH/TO/ADEChallengeData2016 --split_name training
python ade20k_data_convert.py --data_root PATH/TO/ADEChallengeData2016 --split_name validation
```

Example: `meta_grounding_data/ade20k/ade20k_data_convert.sh`. Outputs `ade20k_training_meta.json` / `ade20k_validation_meta.json`.

### LVIS

Download LVIS v1 annotations (`lvis_v1_train.json`, `lvis_v1_val.json`). Images are shared with COCO; `--coco_img_root` must contain `train2017/` and `val2017/`. LVIS splits are not 1-to-1 with COCO splits — image paths are resolved per sample from `coco_url`.

```bash
cd ShowVLA/show-o2/meta_grounding_data/lvis

python lvis_data_convert.py --data_root PATH/TO/LVIS --coco_img_root PATH/TO/COCO --split_name train
python lvis_data_convert.py --data_root PATH/TO/LVIS --coco_img_root PATH/TO/COCO --split_name val
```

Example: `meta_grounding_data/lvis/lvis_data_convert.sh`. Outputs `lvis_train2017_meta.json` / `lvis_val2017_meta.json`. Current mix configs use LVIS + ADE20K (COCO metas are commented out).

Robot-domain grounding (LIBERO / JAKA-clutter / Lumi) is produced separately under `grounding_data_ann/` (e.g. `convert_lumi_grounding.py`). Those metas are listed as `robot_grounding_metas_path` in the mix configs.

## Manipulation Data

Each converter writes per-episode HDF5 (+ preview MP4) and a metainfo JSON. `dataset_name` in the JSON must match the handler registered in `datasets_vla`.

### LIBERO

1. Replay each of the five task suites with `meta_libero_data/libero_data_regen.py` (filters no-ops / failed demos, re-renders 256px, combines main+wrist into `rgb_comb`). Example for one suite: `meta_libero_data/libero_data_regen.sh`.

```bash
cd ShowVLA/show-o2/meta_libero_data

python libero_data_regen.py \
    --libero_task_suite libero_10 \
    --libero_raw_data_dir PATH/TO/LIBERO/libero_10 \
    --libero_target_dir PATH/TO/libero_10_regen
```

Repeat for `libero_spatial`, `libero_object`, `libero_goal`, `libero_90`.

2. Copy the five output files `libero_*_metainfo.json` to `meta_libero_data/split_all/`. Copy `libero_spatial`, `libero_object`, `libero_goal`, `libero_10` to `meta_libero_data/split_spatial_object_goal_10/`.

3. (Optional, used by later mix / stage-2 configs) Build max-length metainfo with `get_metainfo_max.py` and copy into `split_all_max/` and `split_spatial_objec_goal_10_max/`. Example: `meta_libero_data/get_metainfo_max.sh`.

### JAKA

Convert each raw task directory with `meta_jaka_data/convert_jaka.py`. `--crop_main` is task-dependent: `(40, -140, 300, -300)` for most tasks, `(180, -60, 0, -300)` for battery / open-bottle / clutter, `(20, -140, 10, -10)` for `stack_boxes`. Multiple `--data_dir` runs append to the same `{meta_prefix}_metainfo.json`.

```bash
cd ShowVLA/show-o2/meta_jaka_data

python convert_jaka.py \
    --data_dir PATH/TO/JAKA_data/TASK \
    --output_dir PATH/TO/datasets/JAKA/TASK \
    --meta_prefix JAKA \
    --dataset_name JAKA \
    --speed_up 1 \
    --image_stream_offset 1 \
    --crop_main "(40, -140, 300, -300)"
```

- Regular + moveto tasks → `--dataset_name JAKA` (example: `convert_jaka.sh`). Truncated pick/moveto clips use `convert_jaka_pick.py --max_steps N`.
- Clutter put tasks → `--dataset_name JAKA_clutter` (examples: `convert_jaka_clutter.sh`, `convert_jaka_moveto.sh`).

Outputs `JAKA_metainfo.json` / `JAKA_clutter_metainfo.json`.

### CALVIN

Download CALVIN `task_ABC_D` (folders `training/` and `validation/` with `episode_*.npz` and `lang_annotations/`). Converts both splits.

```bash
cd ShowVLA/show-o2/meta_calvin_data

python convert_calvin.py \
    --input_dir PATH/TO/calvin/task_ABC_D \
    --train_dir PATH/TO/datasets/CalvinABC_D/Train \
    --test_dir PATH/TO/datasets/CalvinABC_D/Test \
    --dataset_name Calvin
```

Outputs `Calvin_train_metainfo.json` / `Calvin_test_metainfo.json`. Mix configs use the train split.

### Bridge

Download Bridge RLDS (`bridge_orig/1.0.0`). Requires `tfds-nightly` and `tensorflow` (`pip install tfds-nightly 'tensorflow[and-cuda]'`; if protobuf errors: `pip install 'protobuf>=6.31.1'`).

```bash
cd ShowVLA/show-o2/meta_bridge_data

python convert_bridge.py \
    --input_dir PATH/TO/bridge_orig/1.0.0 \
    --train_dir PATH/TO/datasets/Bridge/Train \
    --val_dir PATH/TO/datasets/Bridge/Val \
    --dataset_name Bridge \
    --split both
```

Then merge train/val metas:

```bash
python merge_meta.py
```

Example: `meta_bridge_data/convert_bridge.sh`. Outputs `train/Bridge_train_metainfo.json`, `val/Bridge_val_metainfo.json`, and `Bridge_all_metainfo.json`.

### DROID

Download DROID RLDS `1.0.1` and the idle-action `keep_ranges_1_0_1.json` (from DROID annotations). Same TensorFlow / TFDS deps as Bridge, plus `mediapy`, `h5py`, `pillow`. Only the `train` split exists. Left-arm and right-arm episodes are written separately.

```bash
cd ShowVLA/show-o2/meta_droid_data

python convert_droid.py \
    --input_dir PATH/TO/droid/1.0.1 \
    --keep_ranges_path PATH/TO/droid/keep_ranges_1_0_1.json \
    --left_dir PATH/TO/datasets/Droid-Left \
    --right_dir PATH/TO/datasets/Droid-Right \
    --left_metainfo ./Droid-Left_metainfo.json \
    --right_metainfo ./Droid-Right_metainfo.json \
    --split train
```

Example: `meta_droid_data/convert_droid.sh`. Outputs `Droid-Left_metainfo.json` / `Droid-Right_metainfo.json`.

### RoboMIND

Download RoboMIND benchmark1_1 HDF5 plus the language annotation JSONs under `static/language_description_annotation_json/`. Two robot embodiments:

**AgileX (3 RGB, bimanual):**

```bash
cd ShowVLA/show-o2/meta_robomind_data

python convert_robomind_agilex.py \
    --data_dir PATH/TO/RoboMIND/benchmark1_1/h5_agilex_3rgb \
    --output_dir PATH/TO/datasets/RoboMIND/benchmark1_1/h5_agilex_3rgb \
    --annotation_json PATH/TO/RoboMIND/static/language_description_annotation_json/h5_agilex_3rgb.json \
    --meta_prefix robomind-agilex \
    --dataset_name robomind-agilex \
    --merge_steps
```

Example: `convert_robomind_agilex.sh`. Output `robomind-agilex_metainfo.json`.

**UR (1 RGB, main view only):**

```bash
python convert_robomind_ur.py \
    --data_dir PATH/TO/RoboMIND/benchmark1_1/h5_ur_1rgb \
    --output_dir PATH/TO/datasets/RoboMIND/benchmark1_1/h5_ur_1rgb \
    --annotation_json PATH/TO/RoboMIND/static/language_description_annotation_json/h5_ur_1rgb.json \
    --meta_prefix robomind-ur \
    --dataset_name robomind-ur
```

Example: `convert_robomind_ur.sh`. Output `robomind-ur_metainfo.json`.

### AGIBOT

AgiBot World Challenge 2025. Sim and real use different raw layouts and converters; `dataset_name` must stay `AGIBOT-HDF5-Sim` / `AGIBOT-HDF5-Real` to match `AGIBOTHDF5Handler`.

**Sim (`Manipulation-SimData`):** per-frame JPGs + `aligned_joints.h5`. EE pose is converted from world to robot-base frame.

```bash
cd ShowVLA/show-o2/meta_agibot_data

python convert_agibot_sim_hdf5.py \
    --task_dir PATH/TO/agibot/Manipulation-SimData \
    --output_dir PATH/TO/datasets/AGIBOT-Sim \
    --metainfo ./AGIBOT-HDF5-Sim_metainfo.json \
    --dataset_name AGIBOT-HDF5-Sim \
    --max_frames 300 \
    --max_downsample_rate 3.5 \
    --min_downsample_rate 2.0
```

Example: `convert_agibot_sim_hdf5.sh`. Output `AGIBOT-HDF5-Sim_metainfo.json`.

**Real (`Manipulation-RealRobot`):** MP4 videos + `proprio_stats.h5` + `task_info/*.json`.

```bash
python convert_agibot_real_hdf5.py \
    --data_root PATH/TO/agibot/Manipulation-RealRobot \
    --output_dir PATH/TO/datasets/AGIBOT-Real \
    --metainfo ./AGIBOT-HDF5-Real_metainfo.json \
    --dataset_name AGIBOT-HDF5-Real \
    --max_frames 400 \
    --max_downsample_rate 3.5 \
    --min_downsample_rate 1.5
```

Example: `convert_agibot_real_hdf5.sh`. Output `AGIBOT-HDF5-Real_metainfo.json`.

### Lumi

Convert each raw Lumi task directory with `meta_lumi_data/convert_lumi.py`. `--dataset_name` must contain `mobile` (chassis velocities). `head_length` / `tail_length` are per-task; `convert_lumi.sh` lists all task dirs with the values used in training.

```bash
cd ShowVLA/show-o2/meta_lumi_data

python convert_lumi.py \
    --data_dir PATH/TO/LumiData/TASK \
    --output_dir PATH/TO/datasets/Lumi/TASK \
    --meta_prefix Lumi \
    --dataset_name Lumi-mobile \
    --speed_up 1 \
    --head_length 20 \
    --tail_length 5 \
    --crop_main "(300, 20, 1060, 720)"
```

Example: `meta_lumi_data/convert_lumi.sh`. Multiple task runs append to `Lumi_metainfo.json`.

## Join Metainfo for Training

Copy (or symlink) the generated metainfo JSONs into a folder under `meta_join_libero_jaka/` and point `training.train_metas_path` at that folder.

| Split folder | Typical use |
| --- | --- |
| `split_all_max_JAKA_Calvin_Bridge_Droid_agilex_ur_AGIBOT` | Stage-1 mix / warmup (full LIBERO + JAKA + CALVIN + Bridge + DROID + RoboMIND + AGIBOT) |
| `split_all_max_JAKA` | Stage-2 full LIBERO + JAKA |
| `split_spatial_object_goal_10_max_JAKA` | Stage-2 continue on LIBERO Spatial/Object/Goal/10 + JAKA |
| `split_all_Lumi_JAKA_Calvin_Bridge_Droid_agilex_ur_AGIBOT` | Mobile mix (adds Lumi) |
| `split_all_Lumi_JAKA` / `split_spatial_object_goal_10_max_JAKA_Lumi` | Mobile stage-2 / continue |

Grounding metas are referenced directly (`grounding_metas_path`, `robot_grounding_metas_path`) and do not need to be copied into the join folder.


# Training

Launch with `accelerate` + DeepSpeed ZeRO-2 (`../accelerate_configs/gpus_deepspeed_zero2.yaml`). Before a run, set `experiment.name` / `experiment.output_dir` in the yaml (and the matching `mkdir` / `ln -s` lines in the shell scripts). Weights are initialized from `show-o2-1.5B-moe`, `Wan2.1_VAE.pth`, `showlab/show-o2-1.5B`, and `2toINF/X-VLA-Pt`.

Default recipe is three stages:

1. **Mix** — visualized (and/or VQA) grounding + VLA, `pred_act: False`.
2. **Future-action** — enable the action head (`pred_act: True`) on full LIBERO + JAKA, after linking the mix checkpoint as `checkpoint-0`.
3. **Continue** — same run dir, target LIBERO suites (Spatial / Object / Goal / 10) + JAKA.

An optional **warmup** between 1 and 2 trains the action head on the full mix VLA set (CALVIN / Bridge / DROID / RoboMIND / AGIBOT included). Stage-2 configs use `resume_from_checkpoint: "latest"`.

`train_vla_mix.sh` / `train_vla_mix_video.sh` chain mix → symlink → stage 2 → continue. The stage-2-only scripts assume the symlink already exists.

## Visualized Grounding + Future Generation (default)

```bash
cd ShowVLA/show-o2
bash train_vla_mix.sh
```

Equivalent stages (edit `MIX_DIR`, `CKPT`, `STAGE2_DIR` to match the yaml `output_dir`):

```bash
# Stage 1 mix: LVIS+ADE20K + robot grounding + full VLA mix, 25k steps
accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla_mix_video.py \
    config=configs/showvla-moe_mix-336x320.yaml

# Link mix ckpt as step 0 of the future-action run
mkdir -p ${STAGE2_DIR}
ln -s ../${MIX_DIR}/checkpoint-${CKPT} ${STAGE2_DIR}/checkpoint-0

# Optional warmup on the full mix VLA set (commented out in train_vla_mix.sh)
# accelerate launch ... train_vla.py config=configs/showvla-moe_future_action-336x320_warmup.yaml

# Stage 2: full LIBERO + JAKA, 10k steps
accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_future_action-336x320.yaml

# Continue: Spatial/Object/Goal/10 + JAKA, resume to 20k
accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_future_action-336x320_cont.yaml
```

Stage 2 / continue can also be run via `train_vla_future_act.sh` and `train_vla_future_act_cont.sh`.

Current yaml defaults: mix dir `showvla-mix_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0825` (`checkpoint-25000`), stage-2 dir `showvla-future_act_Vis_jaka-calvin-bridge-robomind-droid-agibot_lvis0825`. `num_future_imgs: 1`.

## Video Future Generation

Same recipe with `num_future_imgs: 4`:

```bash
bash train_vla_mix_video.sh
```

Configs: `showvla-moe_mix_video-336x320.yaml` → `showvla-moe_video_action-336x320.yaml` → `_cont.yaml` (optional `_warmup.yaml`). Stage-2-only: `train_vla_video_act.sh`. Serve later with `deploy_video.sh` / the video-action config.

## VQA / Combined Grounding (stage 1 only)

Then continue with the future-action stages above after linking the mix checkpoint.

```bash
# VQA-style robot grounding
bash train_vla_mix_vqa.sh          # configs/showvla-moe_mix_vqa-336x320.yaml

# VQA + visualized grounding
bash train_vla_mix_comb.sh         # configs/showvla-moe_mix_combine-336x320.yaml
```

## Mobile (Lumi)

No wrapper script; `pred_mobile_act: True`. After mix (include Lumi grounding / `split_all_Lumi_JAKA_*` metas), link the mix ckpt into the mobile output dir, then:

```bash
accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_mobile_act-336x320_warmup.yaml   # full mix + Lumi, 10k

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_mobile_act-336x320.yaml          # Lumi + JAKA, 18k

accelerate launch --config_file ../accelerate_configs/gpus_deepspeed_zero2.yaml --main_process_port=9999 \
    train_vla.py \
    config=configs/showvla-moe_mobile_act-336x320_cont.yaml     # Spatial/Object/Goal/10 + JAKA + Lumi, 25k
```

# Evaluation

## Serve the model

`deploy.py` loads a checkpoint and starts a FastAPI server. It writes `info.json` (`host`, `port`) under `output_dir` — that file must not already exist.

```bash
cd ShowVLA/show-o2

# Future-action (num_future_imgs=1). Example: deploy.sh
CUDA_VISIBLE_DEVICES=0 python deploy.py \
    config=configs/showvla-moe_future_action-336x320.yaml \
    model_path=${STAGE2_DIR}/checkpoint-${CKPT}/unwrapped_model/pytorch_model.bin \
    output_dir=./${CKPT} \
    device=cuda \
    port=8925 \
    host=0.0.0.0

# Video-action (num_future_imgs=4): deploy_video.sh, config=configs/showvla-moe_video_action-336x320.yaml
```

Use the yaml that matches how the checkpoint was trained (`future_action` vs `video_action`).

## LIBERO

Default: 50 episodes per task, wrist view on the left of `rgb_comb`. Suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`.

```bash
cd ShowVLA/show-o2/evaluation/libero

python libero_client.py \
    --connection_info ../../${CKPT}/info.json \
    --task_suites libero_10 \
    --output_dir ${STAGE2_DIR}/${CKPT}-libero_10 \
    --wrist_at_left True
```

Example: `evaluation/libero/libero_eval.sh`. Repeat per suite / checkpoint. Results are JSON + videos under `--output_dir`.

**LIBERO-plus:** same client; example `evaluation/libero/libero_plus_eval.sh`.

**Remote server:** pass `--server_ip` / `--server_port` instead of `--connection_info`. Example: `evaluation/libero/libero_eval_remote.sh`.
