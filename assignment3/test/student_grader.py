# A custom autograder for this project

################################################################################
# A mini-framework for autograding
################################################################################

import argparse
import pickle
import random
import os, sys
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PARENT_DIR if True else THIS_DIR 
sys.path.append(f'{BASE_DIR}/src/')

class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass

class Tracker(object):
    def __init__(self, questions, maxes, prereqs, mute_output):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.
""".format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        import time
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
          print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (sum(self.points.values()),
            sum([self.maxes[q] for q in self.questions])))

        print("""
Your grades are NOT yet registered. To register your grades, make sure
to follow your instructor's guidelines to receive credit on your project.
""")

    def add_points(self, pts):
        self.points[self.current_question] += pts

TESTS = []
PREREQS = {}
def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)

def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn
    return deco

def parse_options(argv):
    parser = argparse.ArgumentParser(description = 'Run public tests on student code')
    parser.set_defaults(
        edx_output=False,
        gs_output=False,
        graphics=False,
        mute_output=False,
        check_dependencies=False,
        answer_dir='/hint/',
        )
    parser.add_argument('--edx-output', 
                        dest = 'edx_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_argument('--gradescope-output',
                        dest = 'gs_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_argument('--mute',
                        dest = 'mute_output',
                        action = 'store_true',
                        help = 'Mute output from executing tests')
    parser.add_argument('--test',
                        dest = 'test',
                        action = 'store_true',
                        help = 'test implemented answer by TA')
    parser.add_argument('--question', '-q',
                        dest = 'grade_question',
                        default = None,
                        help = 'Grade only one question (e.g. `-q q1`)')
    parser.add_argument('--graphics',
                        dest = 'graphics',
                        action = 'store_true',
                        help = 'Display graphics ?(visualizing your implementation is highly recommended for debugging).')
    parser.add_argument('--check-dependencies',
                        dest = 'check_dependencies',
                        action = 'store_true',
                        help = 'check that numpy and matplotlib are installed')
    parser.add_argument('--answer_dir', 
                        dest = 'answer_dir',
                        help = 'where answers are saved')
    parser.add_argument('--iter-step',
                        dest = 'iter_step',
                        default = 9999999, type = int,
                        help = 'The numbder of iteration steps, should be positive, default 9999999.')
    options, extras = parser.parse_known_args(argv)
    return options

def main():
    options = parse_options(sys.argv[1:])
    if options.check_dependencies:
        check_dependencies()
        return

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, options.mute_output)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker, options)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()

################################################################################
# Tests begin here
################################################################################

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from ir_sim.env import EnvBase
YAML_NAME = 'fastslam_robot'

def check_dependencies():
    pass
  
@test('q0', points=0)
def check_dynamics(tracker, options):
    if options.test:
        from fastslam_robot_answer import RobotFastSLAM
    else:
        from fastslam_robot import RobotFastSLAM
    np.random.seed(16)
    setting = ['q0', 'sim', 'no_measure_resample']

    instructions = [[2.4, 0], [2, 1.57], [2.25, 0], [2, 1.57], [1.3, 0]]
    timepoints   = [3, 1, 2, 1, 2]
    kwargs={'robot_args': {
                's_mode': setting[1], # 'sim', 'pre'
                'e_mode': setting[2], # 'all' (default), 'no_resample', 'no_measure_resample'
                's_R': np.array([0.04, 0, 0.01, 0.01]),
                'e_R': np.array([0.08, 0.02, 0.02, 0.08]),
                'e_Q': np.diag([0.4, 0.2])
                },
            }

    env = EnvBase(f'{BASE_DIR}/src/{YAML_NAME}.yaml', 
                disable_all_plot = not options.graphics, display = options.graphics, 
                custom_robot = RobotFastSLAM, **kwargs)
        
    assert(len(instructions) == len(timepoints))
    k = 0
    next_time = timepoints[k]
    sim_time = np.sum(timepoints)
    
    assert(options.iter_step > 0)
    iter_step = np.min((int(sim_time/env.step_time), options.iter_step))
    # Run simulation
    for i in range(iter_step):
        current_time = env.step_time * i
        if current_time >= next_time:
            k += 1
            next_time += timepoints[k]
        env.step(vel_list=[instructions[k]], )
        env.render(0.1, show_goal=True, show_traj=True)

    if options.graphics:
        env.end(show_traj=True, show_goal=True)

@test('q1', points=5)
def check_fastslam_particles_motion_predict(tracker, options):
    if options.test:
        from fastslam_robot_answer import RobotFastSLAM
    else:
        from fastslam_robot import RobotFastSLAM
    np.random.seed(16)
    setting = ['q1', 'pre', 'no_measure_resample']

    trajectory_true = np.load(f'{BASE_DIR}/{options.answer_dir}/{setting[0]}_trajectory.npy')

    instructions = [[2.4, 0], [2, 1.57], [2.25, 0], [2, 1.57], [1.3, 0]]
    timepoints   = [3, 1, 2, 1, 2]
    kwargs={'robot_args': {
                's_mode': setting[1], # 'sim', 'pre'
                'e_mode': setting[2], # 'all' (default), 'no_resample', 'no_measure_resample'
                's_R': np.array([0.04, 0, 0.01, 0.01]),
                'e_R': np.array([0.08, 0.02, 0.02, 0.08]),
                'e_Q': np.diag([0.4, 0.2])
                },
            }

    env = EnvBase(f'{BASE_DIR}/src/{YAML_NAME}.yaml', 
                disable_all_plot = not options.graphics, display = options.graphics, 
                custom_robot = RobotFastSLAM, **kwargs)
        
    assert(len(instructions) == len(timepoints))
    k = 0
    next_time = timepoints[k]
    sim_time = np.sum(timepoints)
    
    assert(options.iter_step > 0)
    iter_step = np.min((int(sim_time/env.step_time), options.iter_step))
    # Run simulation
    for i in range(iter_step):
        current_time = env.step_time * i
        if current_time >= next_time:
            k += 1
            next_time += timepoints[k]
        env.step(vel_list=[instructions[k]], )
        env.render(0.1, show_goal=True, show_traj=True)

    trajectory = env.robot.e_trajectory
    trajectory = np.concatenate(trajectory, axis = 1)
    error = np.linalg.norm(trajectory-trajectory_true[:, :trajectory.shape[1]] )
    print(f'Trajectory error: {error:3f}')
    if error < 1e-2:
        tracker.add_points(5)

    if options.graphics:
        env.end(show_traj=True, show_goal=True)
            
@test('q2', points=60)
def check_fastslam_particles_measurement_update(tracker, options):
    if options.test:
        from fastslam_robot_answer import RobotFastSLAM
    else:
        from fastslam_robot import RobotFastSLAM
    np.random.seed(16)
    setting = ['q2', 'pre', 'no_resample']

    trajectory_true = np.load(f'{BASE_DIR}/{options.answer_dir}/{setting[0]}_trajectory.npy')

    instructions = [[2.4, 0], [2, 1.57], [2.25, 0], [2, 1.57], [1.3, 0]]
    timepoints   = [3, 1, 2, 1, 2]
    kwargs={'robot_args': {
                's_mode': setting[1], # 'sim', 'pre'
                'e_mode': setting[2], # 'all' (default), 'no_resample', 'no_measure_resample'
                's_R': np.array([0.04, 0, 0.01, 0.01]),
                'e_R': np.array([0.08, 0.02, 0.02, 0.08]),
                'e_Q': np.diag([0.4, 0.2])
                },
            }

    env = EnvBase(f'{BASE_DIR}/src/{YAML_NAME}.yaml', 
                disable_all_plot = not options.graphics, display = options.graphics, 
                custom_robot = RobotFastSLAM, **kwargs)
        
    assert(len(instructions) == len(timepoints))
    k = 0
    next_time = timepoints[k]
    sim_time = np.sum(timepoints)
    
    assert(options.iter_step > 0)
    iter_step = np.min((int(sim_time/env.step_time), options.iter_step))
    # Run simulation
    for i in range(iter_step):
        current_time = env.step_time * i
        if current_time >= next_time:
            k += 1
            next_time += timepoints[k]
        env.step(vel_list=[instructions[k]], )
        env.render(0.1, show_goal=True, show_traj=True)

    trajectory = env.robot.e_trajectory
    trajectory = np.concatenate(trajectory, axis = 1)
    error = np.linalg.norm(trajectory-trajectory_true[:, :trajectory.shape[1]] )
    print(f'Trajectory error: {error:3f}')
    if error < 1e-2:
        tracker.add_points(60)

    if options.graphics:
        env.end(show_traj=True, show_goal=True)

@test('q3', points=35)
def check_particles_resample(tracker, options):
    if options.test:
        from fastslam_robot_answer import RobotFastSLAM
    else:
        from fastslam_robot import RobotFastSLAM
    np.random.seed(16)
    setting = ['q3', 'pre', 'all']

    trajectory_true = np.load(f'{BASE_DIR}/{options.answer_dir}/{setting[0]}_trajectory.npy')

    instructions = [[2.4, 0], [2, 1.57], [2.25, 0], [2, 1.57], [1.3, 0]]
    timepoints   = [3, 1, 2, 1, 2]
    kwargs={'robot_args': {
                's_mode': setting[1], # 'sim', 'pre'
                'e_mode': setting[2], # 'all' (default), 'no_resample', 'no_measure_resample'
                's_R': np.array([0.04, 0, 0.01, 0.01]),
                'e_R': np.array([0.08, 0.02, 0.02, 0.08]),
                'e_Q': np.diag([0.4, 0.2])
                },
            }

    env = EnvBase(f'{BASE_DIR}/src/{YAML_NAME}.yaml', 
                disable_all_plot = not options.graphics, display = options.graphics, 
                custom_robot = RobotFastSLAM, **kwargs)
        
    assert(len(instructions) == len(timepoints))
    k = 0
    next_time = timepoints[k]
    sim_time = np.sum(timepoints)
    
    assert(options.iter_step > 0)
    iter_step = np.min((int(sim_time/env.step_time), options.iter_step))
    # Run simulation
    for i in range(iter_step):
        current_time = env.step_time * i
        if current_time >= next_time:
            k += 1
            next_time += timepoints[k]
        env.step(vel_list=[instructions[k]], )
        env.render(0.1, show_goal=True, show_traj=True)

    trajectory = env.robot.e_trajectory
    trajectory = np.concatenate(trajectory, axis = 1)
    error = np.linalg.norm(trajectory-trajectory_true[:, :trajectory.shape[1]] )
    print(f'Trajectory error: {error:3f}')
    if error < 1e-2:
        tracker.add_points(35)

    if options.graphics:
        env.end(show_traj=True, show_goal=True)

if __name__ == '__main__':
    main()
