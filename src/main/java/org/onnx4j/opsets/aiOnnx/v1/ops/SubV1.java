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
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.OperatorInputs;
import org.onnx4j.opsets.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.tensor.DataType;

/**
 * Sub Operator v1
 * 
 * <p>
 * Performs element-wise binary subtraction (with limited broadcast support).
 * 
 * If necessary the right-hand-side argument will be broadcasted to match the
 * shape of left-hand-side argument. When broadcasting is specified, the second
 * tensor can either be of element size 1 (including a scalar tensor and any
 * tensor with rank equal to or smaller than the first tensor), or having its
 * shape as a contiguous subset of the first tensor's shape. The starting of the
 * mutually equal shape is specified by the argument "axis", and if it is not
 * set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 * 
 * <p>
 * For example, the following tensor shapes are supported (with broadcast=1):
 * 
 * <pre>
 * shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
 * shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
 * shape(A) = (2, 3, 4, 5), shape(B) = (5,)
 * shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
 * shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
 * shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0
 * </pre>
 * 
 * <p>
 * Attribute broadcast=1 needs to be passed to enable broadcasting.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Add-1">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add">ONNX.
 *      Operators.md</a>
 */
public interface SubV1<T_TENSOR> extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Sub";

	class SubInputV1<T_TENSOR> extends OperatorInputs {

		//
		// If set, defines the broadcast dimensions. See doc for details.
		//
		public static final String ATTR_AXIS = "axis";

		//
		// Pass 1 to enable broadcasting.
		//
		public static final String ATTR_BROADCAST = "broadcast";

		//
		// Legacy optimization attribute.
		//
		public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

		private T_TENSOR a;
		private T_TENSOR b;
		private Long axis;
		private Long broadcast;
		private List<Long> consumedInputs;

		public SubInputV1(Node node, Inputs inputs) {
			super(node, inputs);

			Attributes attrs = node.getAttrs();
			Input[] inputArray = inputs.get();

			//
			// axis : int (default is null)
			//
			this.axis = attrs.getAttrValue(ATTR_AXIS, IntAttribute.class, null);

			//
			// boardcast : int (default is 0)
			//
			this.broadcast = attrs.getAttrValue(ATTR_BROADCAST, IntAttribute.class, 0L);

			//
			// Legacy optimization attribute.
			//
			this.consumedInputs = attrs.getAttrValue(ATTR_CONSUMED_INPUTS, IntsAttribute.class, null);

			this.a = inputArray[0].getTensor();
			this.b = inputArray[0].getTensor();
		}

		public T_TENSOR getA() {
			return a;
		}

		public T_TENSOR getB() {
			return b;
		}

		public Long getAxis() {
			return axis;
		}

		public Long getBroadcast() {
			return broadcast;
		}

		public List<Long> getConsumedInputs() {
			return consumedInputs;
		}

	}

	/**
	 * Executes operator
	 * 
	 * @param a
	 *            First operand, should share the type with the second operand.
	 * @param b
	 *            Second operand. With broadcasting can be of smaller size than
	 *            A. If broadcasting is disabled it should be of the same size.
	 * @param axis
	 *            If set, defines the broadcast dimensions. See doc for details.
	 * @param broadcast
	 *            Pass 1 to enable broadcasting.
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @return Result, has same dimensions and type as A
	 */
	public abstract T_TENSOR sub(T_TENSOR a, T_TENSOR b, Long axis, Long broadcast, List<Long> consumedInputs);

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

	@Override
	public default Outputs forward(Node node, Inputs inputs) {
		SubInputV1<T_TENSOR> operatorInput = new SubInputV1<T_TENSOR>(node, inputs);
		return Outputs.wrap(
				node,
				this.sub(
						operatorInput.getA(), 
						operatorInput.getB(), 
						operatorInput.getAxis(), 
						operatorInput.getBroadcast(), 
						operatorInput.getConsumedInputs()
					)
				);
	}

}