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
package org.onnx4j.opsets.aiOnnx.v5.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.aiOnnx.v1.ops.ReshapeV1;
import org.onnx4j.opsets.aiOnnx.v5.AiOnnxOperatorV5;

/**
 * Reshape-5
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Reshape-5
 * @version This version of the operator has been available since version 5 of
 *          the default ONNX operator set.
 *
 */
public interface ReshapeV5<T_TENSOR> extends ReshapeV1<T_TENSOR>, AiOnnxOperatorV5 {

	class ReshapeInputsV5<T_TENSOR> extends ReshapeInputsV1<T_TENSOR> {

		private T_TENSOR shapeTensor;

		public ReshapeInputsV5(Node node, Inputs inputs) {
			super(node, inputs);
			this.shapeTensor = inputs.get()[1].getTensor();
		}

		public T_TENSOR getShapeTensor() {
			return shapeTensor;
		}

	}

	/**
	 * Reshape the input tensor similar to numpy.reshape. It takes a tensor as
	 * input and an argument shape. It outputs the reshaped tensor. At most one
	 * dimension of the new shape can be -1. In this case, the value is inferred
	 * from the size of the tensor and the remaining dimensions. A dimension
	 * could also be 0, in which case the actual dimension value is unchanged
	 * (i.e. taken from the input tensor).
	 * 
	 * @param data
	 *            An input tensor
	 * @param shape
	 *            New shape
	 * @param consumedInputs
	 *            legacy optimization attribute
	 * @return Reshaped data
	 */
	public abstract T_TENSOR reshape(T_TENSOR data, T_TENSOR shape, List<Long> consumedInputs);

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		ReshapeInputsV5<T_TENSOR> operatorInputs = new ReshapeInputsV5<T_TENSOR>(node, inputs);
		return Outputs.wrap(node,
				reshape(operatorInputs.getData(), operatorInputs.getShapeTensor(), operatorInputs.getConsumedInputs()));
	}

}