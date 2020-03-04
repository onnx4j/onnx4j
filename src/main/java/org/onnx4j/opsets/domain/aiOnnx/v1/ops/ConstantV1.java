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
import org.onnx4j.Tensor;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.TensorAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Constant Operator v1
 * 
 * <p>
 * A constant tensor.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Constant-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant">
 *      ONNX.Operators.md</a>
 */
public interface ConstantV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Constant";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	/**
	 * Executes operator
	 * 
	 * @param x0
	 *            The value for the elements of the output tensor.
	 * @return Output tensor containing the same value of the provided tensor.
	 */
	//public abstract T_TENSOR constant(Tensor x0);

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
	class ConstantInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_VALUE = "value";

		protected Field<Tensor> valueField;

		public ConstantInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			//
			// The value for the elements of the output tensor.
			//
			valueField = new AttributeField<Tensor>(super.attrs, ATTR_VALUE, TensorAttribute.class, null, true);
		}

		public TypeConstraint getInputFieldsTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

		public List<T_TENSOR> getInputs() {
			List<T_TENSOR> inputs = new LinkedList<>();
			for (Field<T_TENSOR> field : this.inputFields) {
				inputs.add(field.getData());
			}
			return inputs;
		}

		public Tensor getValue() {
			return valueField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class ConstantOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public ConstantOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}