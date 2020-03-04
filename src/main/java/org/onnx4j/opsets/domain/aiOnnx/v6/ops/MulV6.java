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
package org.onnx4j.opsets.domain.aiOnnx.v6.ops;

import java.util.List;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.MulV1;
import org.onnx4j.opsets.domain.aiOnnx.v6.AiOnnxOperatorV6;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.tensor.DataType;

/**
 * Mul Operator v6
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
 * @version 6
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Mul-6">ONNX.
 *      Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Mul">ONNX.
 *      Operators.md</a>
 */
public interface MulV6 extends MulV1, AiOnnxOperatorV6 {

	public static final String OP_TYPE = "Mul";

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
	 *            First operand, should share the type with the second operand.
	 * @param b
	 *            Second operand. With broadcasting can be of smaller size than
	 *            A. If broadcasting is disabled it should be of the same size.
	 * @param axis
	 *            If set, defines the broadcast dimensions. See doc for details.
	 * @param broadcast
	 *            Pass 1 to enable broadcasting.
	 * @return Result, has same dimensions and type as A
	 */
	//public abstract T_TENSOR mul(T_TENSOR a, T_TENSOR b, Long axis, Long broadcast);

	class MulInputsV6<T_TENSOR> extends MulInputsV1<T_TENSOR> {

		public MulInputsV6(Node node, Inputs inputs) {
			super(node, inputs);
		}

		public List<Long> getConsumedInputs() {
			throw new UnsupportedOperationException(
					String.format("Attribute named \"%s\" has deprecated", ATTR_CONSUMED_INPUTS));
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class MulOutputV6<T_TENSOR> extends MulOutputV1<T_TENSOR> {

		public MulOutputV6(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T;
		}

	}

}