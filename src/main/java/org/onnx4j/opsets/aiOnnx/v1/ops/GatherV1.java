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

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.Outputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.Attributes;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Gather Operator v1
 * 
 * <p>
 * Given data tensor of rank {@literal r >= 1}, and indices tensor of rank q, gather
 * entries of the axis dimension of data (by default outer-most one as axis=0)
 * indexed by indices, and concatenates them in an output tensor of rank q + (r
 * - 1).
 * 
 * <p>
 * Example 1:
 * 
 * <pre>
 * data = [
 *     [1.0, 1.2],
 *     [2.3, 3.4],
 *     [4.5, 5.7],
 * ]
 * indices = [
 *     [0, 1],
 *     [1, 2],
 * ]
 * output = [
 *     [
 *         [1.0, 1.2],
 *         [2.3, 3.4],
 *     ],
 *     [
 *         [2.3, 3.4],
 *         [4.5, 5.7],
 *     ],
 * ]
 * </pre>
 * 
 * <p>
 * Example 2:
 * 
 * <pre>
 * data = [
 *     [1.0, 1.2, 1.9],
 *     [2.3, 3.4, 3.9],
 *     [4.5, 5.7, 5.9],
 * ]
 * indices = [
 *     [0, 2],
 * ]
 * axis = 1,
 * output = [
 *     [
 *         [1.0, 1.9],
 *         [2.3, 3.9],
 *         [4.5, 5.9],
 *     ],
 * ]
 * </pre>
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Reshape-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape">
 *      ONNX.Operators.md</a>
 */
public interface GatherV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Gather";

	public static final String ATTR_AXIS = "axis";

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            Tensor of rank {@literal r >= 1}.
	 * @param indices
	 *            Tensor of int32/int64 indices, of any rank q. All index values
	 *            are expected to be within bounds. It is an error if any of the
	 *            index values are out of bounds.
	 * @param axis
	 *            Which axis to gather on. Negative value means counting
	 *            dimensions from the back. Accepted range is [-r, r-1]
	 * @return Tensor of rank q + (r - 1).
	 */
	public abstract T_TENSOR gather(T_TENSOR data, T_TENSOR indices, Long axis);

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
		private T_TENSOR indices;
		private Long axis;

		public ReshapeInputsV1(Node node, Inputs inputs) {
			super();

			Attributes attrs = node.getAttrs();
			Input[] inputArray = inputs.get();

			this.data = inputArray[0].getTensor();
			this.indices = inputArray[1].getTensor();

			//
			// int (default is 0)
			//
			this.axis = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, 0L);
		}

		public T_TENSOR getData() {
			return data;
		}

		public T_TENSOR getIndices() {
			return indices;
		}

		public Long getAxis() {
			return axis;
		}
	}

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		ReshapeInputsV1<T_TENSOR> operatorInputs = new ReshapeInputsV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(node,
				this.gather(operatorInputs.getData(), operatorInputs.getIndices(), operatorInputs.getAxis()));
	}

}