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
package org.onnx4j.opsets.domain.aiOnnx.v7.ops;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v6.ops.SubV6;
import org.onnx4j.opsets.domain.aiOnnx.v7.AiOnnxOperatorV7;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.tensor.DataType;

/**
 * Sub Operator v7
 * 
 * <p>
 * Performs element-wise binary subtraction (with Numpy-style broadcasting
 * support).
 * 
 * <p>
 * This operator supports multidirectional (i.e., Numpy-style) broadcasting; for
 * more details please check
 * <a href="https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md">the
 * doc</a>.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 7
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Sub-7">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub">ONNX.
 *      Operators.md</a>
 */
public interface SubV7 extends SubV6, AiOnnxOperatorV7 {

	/**
	 * Constrain input and output types to high-precision numeric tensors:
	 * tensor(uint32), tensor(uint64), tensor(int32), tensor(int64),
	 * tensor(float16), tensor(float), tensor(double).
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T = new TypeConstraint(DataType.highPrecisionNumeric());

	/**
	 * Executes operator
	 * 
	 * @param a
	 *            First operand.
	 * @param b
	 *            Second operand.
	 * @return Result, has same element type as two inputs
	 */
	// public abstract T_TENSOR sub(T_TENSOR a, T_TENSOR b);	

	/**
	 * Inputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class SubInputsV7<T_TENSOR> extends SubInputsV6<T_TENSOR> {

		public SubInputsV7(Node node, Inputs inputs) {
			super(node, inputs);
		}

		@Override
		public Long getAxis() {
			throw new UnsupportedOperationException(String.format("Attribute named \"%s\" has deprecated", ATTR_AXIS));
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 * 
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class SubOutputV7<T_TENSOR> extends SubOutputV6<T_TENSOR> {

		public SubOutputV7(T_TENSOR output) {
			super(output);
		}

	}

}