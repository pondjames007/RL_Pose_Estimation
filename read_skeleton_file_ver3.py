import json
import numpy as np
import math
import os
import sys


def read_skeleton_file(filename):
    bodyinfo = []

    with open(filename, 'r') as f:
        framecount = int(f.readline())
        for i in range(framecount):
            bodycount = int(f.readline())
            for b in range(bodycount):
                body = {}
                fileline = f.readline().split(" ")
                body["bodyID"] = int(fileline[0])
                body["clippedEges"] = int(fileline[1])
                body["handLeftConfidence"] = int(fileline[2])
                body["handLeftState"] = int(fileline[3])
                body["handRightConfidence"] = int(fileline[4])
                body["handRightState"] = int(fileline[5])
                body["isRestricted"] = int(fileline[6])
                body["leanX"] = float(fileline[7])
                body["leanY"] = float(fileline[8])
                body["trackingState"] = int(fileline[9])
                body["jointCount"] = int(f.readline())

                joints = []
                for j in range(body["jointCount"]):
                    jointinfo = [float(num) for num in f.readline().split(" ")]
                    joint = {}

                    # 3D location of the joint j
                    joint["x"] = jointinfo[0]
                    joint["y"] = jointinfo[1]
                    joint["z"] = jointinfo[2]

                    # 2D location of the joint j in corresponding depth/IR frame
                    joint["depthX"] = jointinfo[3]
                    joint["depthY"] = jointinfo[4]

                    # 2d location of the joint j in corresponding RGB frame
                    joint["colorX"] = jointinfo[5]
                    joint["colorY"] = jointinfo[6]

                    # The orientation of the joint j
                    joint["orientationW"] = jointinfo[7]
                    joint["orientationX"] = jointinfo[8]
                    joint["orientationY"] = jointinfo[9]
                    joint["orientationZ"] = jointinfo[10]

                    # The tracking state of the joint j
                    joint["trackingState"] = int(jointinfo[11])

                    joints.append(joint)

                body["joints"] = joints

                bodyinfo.append(body)

    return bodyinfo


def convertCoordinate(bodyinfo):
    body_newJoints = []
    for body in bodyinfo:
        r_hip = np.array([body["joints"][17]["x"], body["joints"]
                          [17]["y"], body["joints"][17]["z"]])
        l_hip = np.array([body["joints"][13]["x"], body["joints"]
                          [13]["y"], body["joints"][13]["z"]])
        neck = np.array([body["joints"][3]["x"], body["joints"]
                         [3]["y"], body["joints"][3]["z"]])

        u1 = (r_hip - neck)/np.linalg.norm(r_hip - neck)
        u2 = (l_hip - neck)/np.linalg.norm(l_hip - neck)
        u3 = np.cross(u1, u2)
        u3 /= np.linalg.norm(u3)
        u2 = np.cross(u1, u3)
        u2 /= np.linalg.norm(u2)

        convertMat = np.stack((u3, u2, u1, neck))
        convertMat = np.concatenate(
            (convertMat, np.array([[0, 0, 0, 1]]).T), axis=1)
#         print("\n")
#         print(convertMat.T)
        convertMat = np.linalg.inv(convertMat.T)

        newJoints = []
        for joint in body["joints"]:
            jointPos = np.array([joint["x"], joint["y"], joint["z"], 1])
            newJoint = convertMat.dot(jointPos)/np.linalg.norm(u1)
            newJoint = newJoint[0:3].reshape((1, 3))
            newJoints.append(newJoint)

#         print(newJoints)

        joints_in_one_frame = np.concatenate(newJoints, axis=0)
        # print(joints_in_one_frame)
        body_newJoints.append(joints_in_one_frame)

    return body_newJoints


def calcSpeed(body_newJoints):
    allSpeeds = []  # all frames's speed d
    for i in range(len(body_newJoints)):
        # should we append 0 for the first frame?
        # if i == 0:
        #     speed = np.zeros((len(body_newJoints[i]),3))
        # else:
        if i != 0:
            speed = body_newJoints[i] - body_newJoints[i-1]
            allSpeeds.append(speed)
    return allSpeeds


def calcAcc(allSpeeds):
    allAccs = []
    for i in range(len(allSpeeds)):
        # should we append 0 for the first frame?
        # if i == 0:
        #     acc = np.zeros((len(allSpeeds[i]), 3))
        # else:
        if i != 0:
            acc = allSpeeds[i] - allSpeeds[i-1]
            allAccs.append(acc)
    return allAccs


def exportData(bodyinfo, output_dir, id, isPos):
    ls = [4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 25]
    joints = []
    all_acc = []

    for i in range(len(bodyinfo)):
        if i >= 2:
            frame = []
            acc = []
            for j in range(len(bodyinfo[i]["newPoints"])):
                if j + 1 in ls:
                    v = bodyinfo[i]["newPoints"][j]
                    d = bodyinfo[i]["speed"][j]
                    action = bodyinfo[i]["acceleration"][j]
                    state = [v, d]
                    frame.append(state)
                    action = [action]
                    acc.append(action)
            
            joints.append(frame)
            all_acc.append(acc)
            # print(len(frame))
    # print(joints[0])
    st = np.asarray(joints)
    ac = np.asarray(all_acc)
    if isPos:
        np.savez(output_dir + "clapping_" + str(id) + ".npz", st=st, ac=ac)
    else:
        np.savez(output_dir + "non_clapping_" + str(id) + ".npz", st=st, ac=ac)


def processData(input_file_name, output_dir, id, isPos):
    bodyinfo = read_skeleton_file(input_file_name)
    body_newJoints = convertCoordinate(bodyinfo)
    d = calcSpeed(body_newJoints)
    a = calcAcc(d)
    for i in range(len(bodyinfo)):
        bodyinfo[i]["newPoints"] = body_newJoints[i].tolist()
        if i >= 1:
            bodyinfo[i]["speed"] = d[i - 1].tolist()
        if i >= 2:
            bodyinfo[i]["acceleration"] = a[i - 2].tolist()

    exportData(bodyinfo, output_dir, id, isPos)


if __name__ == "__main__":
    files = []
    input_folder_path = str(sys.argv[1])
    output_folder_path = str(sys.argv[2])
    isPos = str(sys.argv[3]) == "true"
 
    for i in os.listdir(input_folder_path):
        files.append(i)
    id = 0
    for file in files:
        print("Loading file: " + input_folder_path + file)
        processData(input_folder_path + file, output_folder_path, id, isPos)
        id = id + 1
