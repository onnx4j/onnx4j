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
import org.onnx4j.opsets.domain.aiOnnx.v1.AiOnnxOperatorV1;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.OperatorInputs;
import org.onnx4j.opsets.operator.fields.InputField;
import org.onnx4j.opsets.operator.output.SingleOperatorOutputs;
import org.onnx4j.tensor.DataType;

/**
 * MatMul Operator v1
 * 
 * <p>
 * Matrix product that behaves like numpy.matmul:
 * https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 1
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#MatMul-1">
 *      ONNX.Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#MatMul">ONNX
 *      .Operators.md</a>
 */
public interface MatMulV1 extends AiOnnxOperatorV1 {

	public static final String OP_TYPE = "MatMul";

	/**
	 * Constrain input and output types to float tensors.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.floatTypes());

	/**
	 * Executes operator
	 * 
	 * @param x0
	 *            N-dimensional matrix A
	 * @param x1
	 *            N-dimensional matrix B
	 * @return Matrix multiply results from A * B
	 */
	//public abstract T_TENSOR matmul(T_TENSOR x0, T_TENSOR x1);

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
	class MatMulInputsV1<T_TENSOR> extends OperatorInputs<T_TENSOR> {

		protected Field<T_TENSOR> a;

		protected Field<T_TENSOR> b;

		public MatMulInputsV1(Node node, Inputs inputs) {
			super(node, inputs);

			a = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[0]);

			b = new InputField<T_TENSOR>(this, TPYE_CONSTRAINT_T, super.inputArray[1]);
		}

		public T_TENSOR getA() {
			return a.getData();
		}

		public T_TENSOR getB() {
			return b.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward) s
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class MatMulOutputV1<T_TENSOR> extends SingleOperatorOutputs<T_TENSOR> {

		public MatMulOutputV1(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}