/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.onnx4j.opsets.domain.aiOnnx.v12.ops;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.ReduceMaxV1;
import org.onnx4j.opsets.domain.aiOnnx.v11.ops.ReduceMaxV11.ReduceMaxInputsV11;
import org.onnx4j.opsets.domain.aiOnnx.v11.ops.ReduceMaxV11.ReduceMaxOutputV11;
import org.onnx4j.opsets.domain.aiOnnx.v12.AiOnnxOperatorV12;

/**
 * ReduceMax Operator v12
 * 
 * <p>
 * Computes the max of the input tensor's element along the provided axes. The
 * resulted tensor has the same rank as the input if keepdims equal 1. If
 * keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
 * 
 * <p>
 * The above behavior is similar to numpy, with the exception that numpy default
 * keepdims to False instead of True.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 12
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#ReduceMax-12">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax">
 *      ONNX.Operators.md</a>
 */
public interface ReduceMaxV12 extends ReduceMaxV1, AiOnnxOperatorV12 {

	class ReduceMaxInputsV12<T_TENSOR> extends ReduceMaxInputsV11<T_TENSOR> {

		public ReduceMaxInputsV12(Node node, Inputs inputs) {
			super(node, inputs);
		}

	}

	/**
	 * Outputs for operator execution (forward & backward) s
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ReduceMaxOutputV12<T_TENSOR> extends ReduceMaxOutputV11<T_TENSOR> {

		public ReduceMaxOutputV12(T_TENSOR output) {
			super(output);
		}

	}

}