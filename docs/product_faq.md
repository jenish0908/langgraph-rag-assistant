# Northwind AtlasArm — Frequently Asked Questions

## What is the AtlasArm?

The AtlasArm is a 6-axis collaborative robotic arm designed for light industrial
assembly, laboratory automation, and quality-inspection tasks. It is intended to
operate alongside human workers without safety caging, subject to a completed
site risk assessment.

## What is the payload and reach?

The AtlasArm A5 has a maximum payload of 5 kg and a reach of 850 mm. The larger
AtlasArm A12 has a maximum payload of 12 kg and a reach of 1,300 mm.

Payload figures include the weight of the end effector. A 5 kg gripper on an A5
therefore leaves no capacity for the workpiece — size your model accordingly.

## How accurate is it?

Repeatability is +/- 0.03 mm for the A5 and +/- 0.05 mm for the A12, measured
per ISO 9283 at full extension under rated load. Absolute positional accuracy is
lower than repeatability and depends on calibration; expect +/- 0.2 mm after a
standard factory calibration.

## What are the power requirements?

The A5 runs on single-phase 100-240V AC, 50/60 Hz, drawing a maximum of 350 W.
The A12 requires 200-240V AC only and draws up to 800 W. Neither model requires
three-phase power.

Peak draw during rapid acceleration can briefly reach 1.5x the rated figure.
Size circuit protection accordingly.

## How is it programmed?

Three options are supported:

1. **Hand-guiding.** Physically move the arm through the desired path while the
   controller records waypoints. Requires no programming knowledge and is the
   fastest route for simple pick-and-place tasks.
2. **AtlasStudio.** A visual, block-based editor running in the browser. Suitable
   for conditional logic, I/O handling, and multi-station sequences.
3. **Python SDK.** Full programmatic control over motion, I/O, and the vision
   subsystem. Requires Python 3.9 or later. This is the only option that exposes
   real-time force-torque feedback.

ROS 2 Humble and Jazzy are supported through a community-maintained driver. The
driver is not covered by the commercial support agreement.

## What safety certifications does it hold?

The AtlasArm is certified to ISO 10218-1 and ISO/TS 15066 for collaborative
operation. It carries CE marking and is UL listed for North America.

Collaborative operation still requires a site-specific risk assessment under
ISO 12100. Certification of the arm does not certify your application. Sharp
end effectors, or payloads that could become projectiles, will typically require
additional guarding regardless of the arm's rating.

## What is the warranty?

Hardware is warranted for 24 months from the date of delivery. The warranty
covers manufacturing defects and premature mechanical wear. It does not cover
collision damage, water ingress, use outside the rated environmental envelope,
or damage caused by third-party end effectors.

Extended warranty to 48 months is available at purchase for 18% of the list
price. It cannot be purchased retroactively.

## What is the expected service life?

Mechanical service life is rated at 35,000 operating hours. The harmonic drives
in joints 1 through 3 are the typical wear-limited components and are designed
to be field-replaceable.

Recommended preventive maintenance is annual, or every 4,000 operating hours,
whichever comes first. Skipping scheduled maintenance voids the warranty.

## What environmental conditions are supported?

Operating temperature is 5 to 40 degrees Celsius with non-condensing humidity
between 20% and 80%. The standard model is rated IP54. An IP67-rated washdown
variant of the A5 is available for food and pharmaceutical applications at a 25%
price premium.

The arm must not be operated in explosive atmospheres. There is no ATEX-rated
variant, and none is planned.

## What support is included?

All commercial purchases include 12 months of standard support: email response
within one business day, and access to the knowledge base and firmware updates.

Premium support adds 24/7 phone coverage, a 4-hour response target, and advance
hardware replacement. It is priced at 12% of list price annually.

Firmware updates are free for the life of the product. Major version upgrades
that require new hardware are not covered.
