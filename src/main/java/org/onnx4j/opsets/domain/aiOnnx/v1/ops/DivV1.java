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
package org.onnx4j.opsets.domain.aiOnnx.v1.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Div Operator v1
 * 
 * <p>
 * Performs element-wise binary division (with limited broadcast support).
 * 
 * If necessary the right-hand-side argument will be broadcasted to match the
 * shape of left-hand-side argument. When broadcasting is specified, the second
 * tensor can either be of element size 1 (including a scalar tensor and any
 * tensor with rank equal to or smaller than the first tensor), or having its
 * shape as a contiguous subset of the first tensor's shape. The starting of the
 * mutually equal shape is specified by the argument "axis", and if it is not
 * set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 * 
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
 * Attribute broadcast=1 needs to be passed to enable broadcasting.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Div-1">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div">ONNX.
 *      Operators.md</a>
 */
public interface DivV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Div";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

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

	/**
	 * Executes operator
	 * 
	 * @param a
	 *            First operand, should share the type with the second operand.
	 * @param b
	 *            Second operand. With broadcasting can be of smaller size than
	 *            A. If broadcasting is disabled it should be of the same size.
	 * @return Result, has same dimensions and type as A
	 */
	//public abstract T_TENSOR div(T_TENSOR a, T_TENSOR b, Long axis, Long broadcast, List<Long> consumedInputs);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	class DivInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

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

		protected Field<T_TENSOR> aField;

		protected Field<T_TENSOR> bField;

		protected Field<Long> axisField;

		protected Field<Long> broadcastField;

		protected Field<List<Long>> consumedInputsField;

		public DivInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			aField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);
			bField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[1]);

			axisField = new AttributeField<Long>(super.attrs, ATTR_AXIS, IntAttribute.class, null, false);
			broadcastField = new AttributeField<Long>(super.attrs, ATTR_BROADCAST, IntAttribute.class, 0L, false);
			consumedInputsField = new AttributeField<List<Long>>(super.attrs, ATTR_CONSUMED_INPUTS, IntsAttribute.class,
					null, false);
		}

		public T_TENSOR getA() {
			return aField.getData();
		}

		public T_TENSOR getB() {
			return bField.getData();
		}

		public Long getAxis() {
			return axisField.getData();
		}

		public Long getBroadcast() {
			return broadcastField.getData();
		}

		public List<Long> getConsumedInputs() {
			return consumedInputsField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class DivOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public DivOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}