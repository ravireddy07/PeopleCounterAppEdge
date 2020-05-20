import os
import sys
import time
import socket
import json
import cv2
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

HostName = socket.gethostname()
ipAddress = socket.gethostbyname(HostName)
mqtt_host = ipAddress
mqtt_port = 3001
mqtt_break = 60

def outputSSD(frame, result):
    current_count = 0
    for obj in result[0][0]:
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count

def performance_counts(perf_count):
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))

def buildd():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def connect_mqtt():
    client = mqtt.Client()
    client.connect(mqtt_host, mqtt_port, mqtt_break)
    return client


def infer_on_stream(args, client):
    single_image_mode = False
    cur_request_id = 0
    last_count = 0
    total_count = 0
    start_time = 0
    infer_network = Network()
    prob_threshold = args.prob_threshold
    infer_network = Network()
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1,
                                          cur_request_id, 
                                          args.cpu_extension)[1]
    
    
    if args.input == 'CAM':
        input_stream = 0

    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR!! Unable to open Video Source")
    global initial_w, initial_h, prob_threshold
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)

    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        image = cv2.resize(frame, (w, h))
        
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        inf_start = time.time()
        
        infer_network.exec_net(cur_request_id, image)

        if infer_network.wait(cur_request_id) == 0:
            
            det_time = time.time() - inf_start
            
            result = infer_network.get_output(cur_request_id)
            perf_count = infer_network.performance_counter(cur_request_id)

            frame, current_count = outputSSD(frame, result)
            
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))

            if current_count < last_count:
                duration = int(time.time() - start_time)
                client.publish("person/duration", json.dumps({"duration": duration}))

            client.publish("person", json.dumps({"count": current_count}))
            last_count = current_count

        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()

        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

def main():
    args = buildd().parse_args()
    client = connect_mqtt()
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
