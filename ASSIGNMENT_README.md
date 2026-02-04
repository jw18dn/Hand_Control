# Robotic Hand Control Assignment - Reflex Robotics

## What you'll be doing
You'll be working with a robotic hand model in MuJoCo and making it track real hand motion data (hand retargetting). We have CSV files with fingertip positions from motion capture, and your job is to make the simulated robot hand follow these movements.

## What we're giving you

1. **Hand model**: `xml/7DHand.xml` - A MuJoCo robot hand with 13 joints
2. **Motion data**: Three CSV files with fingertip positions obtained from motion capture
   - `hand_tracking_data_simple.csv` - General movements
   - `hand_tracking_data_pinch.csv` - Clean pinching motions
   - `hand_tracking_data_pinch_complex.csv` - Complex pinching with noisy measurements

The CSV has timestamps (in milliseconds) and positions/orientations for the palm, thumb, index, and middle fingertips.

## The Assignment

### Part 1: Get the simulation running
Set up a C++ program that:
- Loads the hand model in MuJoCo
- Runs the simulation at 1000Hz (1ms timestep) in real-time
- Shows the hand on screen (GUI)

### Part 2: Make the fingers track the CSV data 
Now the fun part:
- Read the CSV tracking data (start with the `hand_tracking_data_simple.csv` then use `hand_tracking_data_pinch.csv`)
- Make the robot fingertips follow the recorded positions
- All three fingers (thumb, index, middle) should track simultaneously along with the palm orientation

### Part 3: Handle joint coupling
Real robotic hands often have mechanical coupling between joints (tendons, linkages, etc.). Our robot has coupling between the last two joints of the index and middle fingers. 

Your controller needs to handle this constraint:
- For index finger: if JI_L2 rotates by angle θ, then JI_L3 must rotate by ratio × θ
- For middle finger: if JP_L2 rotates by angle θ, then JP_L3 must rotate by ratio × θ
- The coupling ratio is a parameter that is defined by the hardware team

This means you can't control these joints independently anymore - when tracking fingertip positions, update your controller so that it can respect this constraints when deciding the commands. Make the ratio configurable so we can test different values. 

### Part 4: Handle pinching 
The pinch CSVs have the thumb touching the index finger or the middle finger and sometimes both. All this can happen with palm rotating, Your controller needs to:
- Detect when fingers are pinching (within ~10mm)
- Keep the robot fingers in contact (or pinching configuration) during the pinch
- Still track the overall motion while maintaining the pinch

**Important**: In `hand_tracking_data_pinch_complex.csv`, the human is pinching but the tracking measurements won't always show the fingers as being very close (due to poor calibration and device estimation.). Your controller should be robust enough to detect the pinch intent and maintain contact between the fingers even when the raw measurements say the fingers are 15-20mm apart.

## How far should you go?
**Get as far as you can, but don't stress about completing everything.** We expect candidates to atleast complete Parts 1, 2. Solving parts 3 & 4 is what will make us very excited about you! 

## Libraries and Tools

**Use whatever libraries help you solve the problem!** Want to use Eigen for linear algebra? Pinocchio for Robot Math? OSQP for optimization? Go for it. We care about solutions, not restrictions. Just make sure to document your dependencies so we can build your code.

## AI Tools and Reimbursement

**You're encouraged to use AI tools!** Use GPT-5, Claude, Claude Code, GitHub Copilot, or whatever helps you work effectively. This is how we actually work, and we want to see how you leverage these tools.

**Reflex Robotics will reimburse up to $200 USD for any AI tool subscriptions or API costs** you use during this assignment. Just include your receipts with your submission.

## What to submit

Send us:
```
your_name_submission/
├── src/           (your C++ code)
├── third_party/         
├── build/         
├── CMakeLists.txt 
├── README.md      (how to build/run, your approach, any issues)
└── results/       (videos showing it working)
```
Your package should be more or less self contained except for system wide packages. It should be easy for us to build and test

Include videos of:
1. The simulation running at 1kHz
2. Fingers tracking the general motion CSV
3. The pinching behavior working
4. Joint coupling working (show with different ratio values)

Also write a short README explaining:
- How to build and run your code
- Your approach
- What works and what doesn't
- Any assumptions you made

## Grading

We care most about working code, but reasonable code hygiene matters too - we need to be able to understand and potentially extend your solution. Get it functional first, then clean it up to a professional standard. No need for over-engineering, just write code you'd be comfortable submitting as a PR at work.

Make sure to write down your thought process, design choices and the assumptions you make. We care about how you approach the problem and why you make these decisions. 

## Questions?

If something's unclear, just ask! Document any assumptions in your README.

Good luck!