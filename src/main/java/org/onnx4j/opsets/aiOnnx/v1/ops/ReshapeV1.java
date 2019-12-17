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
package org.onnx4j.opsets.aiOnnx.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Div-1
 * 
 * @author HarryLee
 * @see https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Div-1
 * @version This version of the operator has been available since version 1 of
 *          the default ONNX operator set.
 *
 */
public interface ReshapeV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Reshape";

	//
	// New shape
	//
	public static final String ATTR_SHAPE = "shape";

	//
	// Legacy optimization attribute.
	//
	public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

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
	public abstract T_TENSOR reshape(T_TENSOR data, List<Long> shape, List<Long> consumedInputs);

	@Override
	public default DataType[] getTypeConstraints() {
		return DataType.floatTypes();
	}

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	class ReshapeInputsV1<T_TENSOR> {
		private T_TENSOR data;
		private List<Long> shape;
		private List<Long> consumedInputs;

		public ReshapeInputsV1(Node node, Inputs inputs) {
			super();

			Attributes attrs = node.getAttrs();
			Input[] inputArray = inputs.get();

			this.data = inputArray[0].getTensor();
			this.shape = attrs.getAttrValue(ATTR_SHAPE, IntsAttribute.class, null);
			this.consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);
		}

		public T_TENSOR getData() {
			return data;
		}

		public List<Long> getShape() {
			return shape;
		}

		public List<Long> getConsumedInputs() {
			return consumedInputs;
		}
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		ReshapeInputsV1<T_TENSOR> operatorInputs = new ReshapeInputsV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node,
				this.reshape(operatorInputs.getData(), operatorInputs.getShape(), operatorInputs.getConsumedInputs()));
	}

}