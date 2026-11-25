# Cross-task lessons file

The orchestrator maintains this file. It contains only rules confirmed by native evaluation; rules derived from
demonstrations are marked separately. Every new candidate prompt refers to it, so it is the starting point of every
new candidate. When adding a rule, name the task(s) it was confirmed on.

## Motion and inverse kinematics

1. A generic IK solver that is not seeded with the current joints returns arbitrary arm-configuration branches, often
   off by 2π, and the URDF joint limits (±10 rad) admit solutions that are unreachable in simulation. Remedy: among the
   2π-equivalent solutions take the one closest to the current joints; reject any segment with a single-joint jump
   > 1.5 rad; for neighbouring poses such as grasp and lift, start from the hover joints. (general_pickup,
   classify_objects, organize_table)
2. Planning and execution share one URDF damped-least-squares IK chain seeded at the current joints and constrained to
   the real joint limits (from the robot model, not scene data: j2/j3 ∈ [0, 180°], j4/j5 ∈ ±90°). This removes
   branch jumps, and planning drops from hundreds of seconds to 1–15 s. Precompute the FK chain. (pack_objects_into_box,
   classify_objects)
3. Check IK for every pose that is actually executed (pre-grasp, grasp, lift, transit, release, retreat), and use the
   same check to choose the base/stacking point; when IK fails, retry with a lower pre-grasp/transit height.
   (stack_blocks)
4. The two arms' reachable regions barely overlap; a vertical hand only reaches 0.2–0.35 m from its base. For distant
   objects, grasp tilted by 0/15/30/45° about the gripper closing axis (the closing axis must be perpendicular to the
   radial direction, otherwise an oblique pinch slips); if that still fails, relay via the table to the other hand.
   (stack_blocks, stack_blocks_by_language, general_pickup)
5. Placing into a container far from the base: tilt the hand forward along the arm's radial direction by 0–65° for
   the release, and check the waypoints and the release point first. (classify_objects)
6. After a failure, do not open the gripper in mid-air; put the object back on the table first. (classify_objects)
7. If the straight lift after release has no IK solution (release at the front edge), back out along the already
   checked approach path rather than going home in joint space directly from the release pose. (stack_blocks)

## Perception

8. Move both arms out of the head camera's view before observing, otherwise objects are missed. (classify_objects)
9. Depth alone often suffices: table-plane histogram + height bands + removing hand points via encoder forward
   kinematics + grid clustering; the minimum-area rectangle of the top face gives centre and orientation.
   (stack_blocks, stack_blocks_by_language)
10. Once an object is covered, the head camera only sees its front face and the centre is off by about 1.2 cm; use the
    position from the last observation in which the top face was visible. (stack_blocks_by_language)
11. Objects in contact merge into one depth-connected component; split them using their known size. (stack_bowls)
12. Grasping a narrow part: slice the point cloud along the main axis into 2 cm slices and take a slice narrower than
    the gripper opening; rank top-down grasps by the number of clutter points inside the hand's collision envelope.
    (general_pickup)
13. Front/back of an elongated object such as a shoe: decide from depth geometry (e.g. the heel collar is the highest
    point) rather than from appearance. (pack_objects_into_box)

## Grasping and placing

14. Bowls: tilt the wrist outward by 25° to pinch the bowl wall; a vertical pinch on the rim tilts the bowl strongly.
    If the in-hand tilt exceeds 38°, put it down and re-grasp. (stack_bowls)
15. Pushing objects: push with the gripper open for two-point contact to avoid rotating the object; push speed about
    6 mm per action. (organize_table)
16. Grasp tall objects above their centre of mass (e.g. a 4 cm cube at mid-height + 4 mm, ≥ 1 cm below the top);
    below the centre of mass the object swings about the closing axis and lands offset and tilted. (stack_blocks)
17. During transport, first lift the hand vertically, translate above the target, then tilt forward and lower;
    rotating towards the release pose while moving makes hanging objects scrape the container rim and fall.
    (classify_objects)
18. Placing into containers: enter low from the near box wall, tilting by only 10–20° for the final slide; a
    clearance computation that counts flaps/lid pushes the pose into an IK-unreachable region. (pack_objects_into_box)
19. Releasing inside a box: open only to grasp width + 1 cm → lift vertically above the rim → open fully → retreat,
    so the box walls are not spread, the box is not pushed, and the object is not dragged back out. For collision
    checks use the full point cloud that moves rigidly with the box (including flaps), not a cropped copy.
    (pack_objects_into_box)
20. Dragging a light box: straddle the front wall top with the gripper open (about 3.5 cm overlap), close, pull along
    the wall normal at about 6 mm per action, then release; pushing on the inner wall is IK-infeasible.
    (pack_objects_into_box)
21. The native gripper control limits the target opening to "measured opening ± 0.2 stroke" and the gripping force
    cannot be raised by command, so long heavy objects (wrench, hammer) can only be held by grasp location and smooth
    acceleration. (store_tools_in_toolbox)
22. Slender tools: slide the tool along the table to the outside of the box wall, lift it over the wall at the lowest
    feasible height and lower it vertically, instead of carrying it through the air and releasing above the slot.
    (store_tools_in_toolbox)

## Verification and efficiency

23. Empty-grasp detection is visual: after lifting, check in the wrist image that the object is between the jaws, and
    check in head depth whether an object is still at the original location.
24. Parse instructions with regular expressions; no open-ended path search; 3–8 minutes per episode is achievable.
25. Table relay: rank hand-over points by "free space, near the source arm's side", and use one only if the receiving
    arm has a complete path from the hand-over point to the target and the source arm can carry the object there.
    (classify_objects_by_language)
