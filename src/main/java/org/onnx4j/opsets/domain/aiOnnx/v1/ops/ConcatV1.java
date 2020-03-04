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

import java.util.LinkedList;
import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.Inputs.Input;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Concat Operator v1
 * 
 * <p>
 * Concatenate a list of tensors into a single tensor
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Concat-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Concat">ONNX
 *      .Operators.md</a>
 */
public interface ConcatV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Concat";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	/**
	 * Executes operator
	 * 
	 * @param inputs
	 *            List of tensors for concatenation
	 * @return Concatenated tensor
	 */
	//public abstract T_TENSOR concat(List<T_TENSOR> inputs, Long axis);

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	/**
	 * Inputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ConcatInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_AXIS = "axis";

		protected List<Field<T_TENSOR>> inputFields;

		protected Field<Long> axisField;

		public ConcatInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			this.inputFields = new LinkedList<>();
			for (Input input : super.inputArray) {
				inputFields.add(new InputField<T_TENSOR>(this, getInputFieldsTypeConstraint(), input));
			}

			axisField = new AttributeField<Long>(super.attrs, ATTR_AXIS, IntAttribute.class, 1L, true);
		}

		public TypeConstraint getInputFieldsTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

		/**
		 * @return List of tensors for concatenation
		 */
		public List<T_TENSOR> getInputs() {
			List<T_TENSOR> inputs = new LinkedList<>();
			for (Field<T_TENSOR> field : this.inputFields) {
				inputs.add(field.getData());
			}
			return inputs;
		}

		/**
		 * @return Which axis to concat on. Default value is 1.
		 */
		public Long getAxis() {
			return axisField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ConcatOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		/**
		 * @param output Concatenated tensor
		 */
		public ConcatOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}