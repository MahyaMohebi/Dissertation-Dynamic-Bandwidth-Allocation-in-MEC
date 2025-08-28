#!/usr/bin/env python3
"""Export a Keras .h5 model to ONNX.
Usage:
  python ml/export_onnx.py --model ml/models/checkpoints/model.h5 --out ml/models/final/bandwidth_lstm.onnx
"""
import argparse, os, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', default='ml/models/final/bandwidth_lstm.onnx')
    ap.add_argument('--opset', type=int, default=13)
    args = ap.parse_args()

    try:
        import tensorflow as tf
        import tf2onnx
        from tf2onnx import tf_loader
    except Exception as e:
        print('ERROR: tf2onnx not installed. Install with: pip install tf2onnx', file=sys.stderr)
        return 2

    model = tf.keras.models.load_model(args.model, compile=False)
    spec = (tf.TensorSpec((None, model.input_shape[1], 1), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=args.opset, output_path=args.out)
    print('Exported ONNX to', args.out)

if __name__ == '__main__':
    raise SystemExit(main())
