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

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * Abs Operator v1
 * 
 * <p>
 * Absolute takes one input data (Tensor) and produces one output data (Tensor)
 * where the absolute is, y = abs(x), is applied to the tensor elementwise.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Abs-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Abs">
 *      ONNX.Operators.md</a>
 */
public interface AbsV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "Abs";

	/**
	 * Constrain input types. Absing from complex is not supported.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T1 = new TypeConstraint(
			DataType.allTypesWithoutComplex());

	/**
	 * Constrain input types. Absing from complex is not supported.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T2 = new TypeConstraint(
			DataType.allTypesWithoutComplex());

	/**
	 * Executes operator
	 * 
	 * @param x
	 *            Input tensor
	 * @return Output tensor
	 */
	/*public T_TENSOR abs(T_TENSOR x);*/

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
	 * @param <T_TENSOR> The backend tensor object.
	 */
	class AbsInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		protected InputField<T_TENSOR> x = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T1, super.inputArray[0]);

		public AbsInputsV1(Node node, Inputs inputs) {
			super(node, inputs);
		}

		public T_TENSOR getX() {
			return x.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward) 
	 *
	 * @param <T_TENSOR> The backend tensor object.
	 */
	class AbsOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public AbsOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T2;
		}

	}

}