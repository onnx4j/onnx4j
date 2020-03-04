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
import org.onnx4j.model.graph.node.attributes.FloatAttribute;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.model.graph.node.attributes.IntsAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.OperatorOutputs;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.fields.OutputField;
import org.onnx4j.opsets.operator.output.MultiOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Dropout Operator v1
 * 
 * <p>
 * Dropout takes one input data (Tensor) and produces two Tensor outputs, output
 * (Tensor) and mask (Tensor). Depending on whether it is in test mode or not,
 * the output Y will either be a random dropout, or a simple copy of the input.
 * Note that our implementation of Dropout does scaling in the training phase,
 * so during testing nothing needs to be done.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Dropout-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Dropout">
 *      ONNX.Operators.md</a>
 */
public interface DropoutV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Dropout";

	/**
	 * Constrain input and output types to all numeric tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.numericTypes());

	@Override
	public default OperatorStatus getStatus() {
		return OperatorStatus.STABLE;
	}

	@Override
	public default String getOpType() {
		return OP_TYPE;
	}

	/*@Override
	public default OperatorInputs<T_TENSOR> asOperatorInputs(Node node, Inputs inputs) {
		return new DropoutInputsV1<T_TENSOR>(node, inputs);
	}*/

	/**
	 * Executes operator
	 * 
	 * @param data
	 *            The input data as Tensor.
	 * @param isTest
	 *            (int, default 0) if nonzero, run dropout in test mode where
	 *            the output is simply Y = X.
	 * @param ratio
	 *            (float, default 0.5) the ratio of random dropout
	 * @param consumedInputs
	 *            legacy optimization attribute.
	 * @return output : T The output. mask (optional) : T The output mask. If
	 *         is_test is nonzero, this output is not filled.
	 */
	class DropoutInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		public static final String ATTR_RATIO = "ratio";

		public static final String ATTR_IS_TEST = "is_test";

		public static final String ATTR_CONSUMED_INPUTS = "consumed_inputs";

		protected Field<T_TENSOR> dataField;

		protected Field<Float> ratioField;

		protected Field<Long> isTestField;

		protected Field<List<Long>> consumedInputsField;

		public DropoutInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			this.dataField = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);
			this.ratioField = new AttributeField<Float>(super.attrs, ATTR_RATIO, FloatAttribute.class, 0.5f, false);
			this.isTestField = new AttributeField<Long>(super.attrs, ATTR_IS_TEST, IntAttribute.class, 0L, false);
			this.consumedInputsField = new AttributeField<List<Long>>(super.attrs, ATTR_CONSUMED_INPUTS,
					IntsAttribute.class, null, false);
		}

		public T_TENSOR getData() {
			return dataField.getData();
		}

		public Float getRatio() {
			return ratioField.getData();
		}

		public Boolean isTest() {
			return isTestField.getData() != 0;
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
	class DropoutOutputV1<T_TENSOR> extends MultiOperatorOutputs<T_TENSOR> {

		protected Field<T_TENSOR> dataField;

		protected Field<T_TENSOR> maskField;

		public DropoutOutputV1(OperatorOutputs<T_TENSOR> operatorOutputs) {
			//
			// Attention order of fields initialization
			//
			this.dataField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, operatorOutputs.get(0), false);
			this.maskField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, operatorOutputs.get(1), true);
		}

		public DropoutOutputV1(T_TENSOR data) {
			this(data, null);
		}

		public DropoutOutputV1(T_TENSOR data, T_TENSOR mask) {
			//
			// Attention order of fields initialization
			//
			this.dataField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, data, false);
			this.maskField = new OutputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, mask, true);
		}

	}

}