# from isaaclab.managers import RewardTermCfg as RewTerm

# @configclass
# class RewardsCfg:
#     """Reward terms for bimanual cube manipulation."""

#     # 1) Encourage each hand to approach its cube
#     reach_left_cube = RewTerm(
#         func=manip_mdp.distance_to_target,
#         params={
#             "eef_link_name": "left_wrist_yaw_link",
#             "target_asset_cfg": SceneEntityCfg("left_object"),
#             "distance_scale": 1.0,
#             "use_negative_dist": True,   # reward = -distance
#         },
#         weight=1.0,
#     )

#     reach_right_cube = RewTerm(
#         func=manip_mdp.distance_to_target,
#         params={
#             "eef_link_name": "right_wrist_yaw_link",
#             "target_asset_cfg": SceneEntityCfg("right_object"),
#             "distance_scale": 1.0,
#             "use_negative_dist": True,
#         },
#         weight=1.0,
#     )

#     # 2) Reward lifting cubes up from the table
#     lift_left_cube = RewTerm(
#         func=manip_mdp.asset_height,
#         params={
#             "asset_cfg": SceneEntityCfg("left_object"),
#             "reference_height": 0.70,   # table height
#         },
#         weight=2.0,
#     )

#     lift_right_cube = RewTerm(
#         func=manip_mdp.asset_height,
#         params={
#             "asset_cfg": SceneEntityCfg("right_object"),
#             "reference_height": 0.70,
#         },
#         weight=2.0,
#     )

#     # 3) Small penalty on large joint motions
#     action_smoothness = RewTerm(
#         func=base_mdp.action_rate_l2,
#         weight=-1e-3,
#     )
