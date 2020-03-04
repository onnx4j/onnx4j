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
 * Mul Operator v1
 * 
 * <p>
 * Performs element-wise binary multiplication (with limited broadcast support).
 * 
 * If necessary the right-hand-side argument will be broadcasted to match the
 * shape of left-hand-side argument. When broadcasting is specified, the second
 * tensor can either be of element size 1 (including a scalar tensor and any
 * tensor with rank equal to or smaller than the first tensor), or having its
 * shape as a contiguous subset of the first tensor's shape. The starting of the
 * mutually equal shape is specified by the argument "axis", and if it is not
 * set, suffix matching is assumed. 1-dim expansion doesn't work yet.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Mul-1">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul">ONNX.
 *      Operators.md</a>
 */
public interface MulV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Mul";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

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
	//public abstract T_TENSOR mul(T_TENSOR a, T_TENSOR b, Long axis, Long broadcast, List<Long> consumedInputs);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}
	
	class MulInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

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

		protected Field<T_TENSOR> aField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

		protected Field<T_TENSOR> bField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[1]);

		protected Field<Long> axisField = new AttributeField<Long>(super.attrs, ATTR_AXIS, IntAttribute.class, null,
				false);

		protected Field<Long> broadcastField = new AttributeField<Long>(super.attrs, ATTR_BROADCAST, IntAttribute.class,
				0L, false);

		protected Field<List<Long>> consumedInputsField = new AttributeField<List<Long>>(super.attrs,
				ATTR_CONSUMED_INPUTS, IntsAttribute.class, null, false);

		public MulInputsV1(Node node, Inputs inputs) {
			super(node, inputs);
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
	class MulOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public MulOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}