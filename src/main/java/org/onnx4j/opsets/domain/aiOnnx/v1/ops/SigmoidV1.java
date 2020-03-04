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
 * Sigmoid Operator v1
 * 
 * <p>
 * Sigmoid takes one input data (Tensor) and produces one output data (Tensor)
 * where the sigmoid function, y = 1 / (1 + exp(-x)), is applied to the tensor
 * elementwise.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Sigmoid-1">
 *      ONNX. Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid">
 *      ONNX. Operators.md</a>
 */
public interface SigmoidV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Sigmoid";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	/**
	 * Executes operator
	 * 
	 * @param x
	 *            Input tensor.
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @return Output tensor.
	 */
	//public abstract T_TENSOR sigmoid(T_TENSOR x, List<Long> consumedInputs);

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
	class SigmoidInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

		private Field<List<Long>> consumedInputsField;

		private Field<T_TENSOR> xField;

		public SigmoidInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			xField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			//
			// Legacy optimization attribute.
			//
			consumedInputsField = new AttributeField<List<Long>>(super.attrs, ATTR_CONSUMED_INPUTS, IntsAttribute.class,
					null, false);
		}

		public T_TENSOR getX() {
			return xField.getData();
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
	class SigmoidOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public SigmoidOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}