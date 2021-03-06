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
package org.onnx4j.opsets.domain.aiOnnx.v9.ops;

import org.onnx4j.Inputs;
import org.onnx4j.model.graph.Node;
import org.onnx4j.model.graph.node.attributes.IntAttribute;
import org.onnx4j.opsets.domain.aiOnnx.v1.ops.CastV1;
import org.onnx4j.opsets.domain.aiOnnx.v6.ops.CastV6;
import org.onnx4j.opsets.domain.aiOnnx.v9.AiOnnxOperatorV9;
import org.onnx4j.opsets.operator.Field;
import org.onnx4j.opsets.operator.Field.TypeConstraint;
import org.onnx4j.opsets.operator.fields.AttributeField;
import org.onnx4j.tensor.DataType;

/**
 * Cast Operator v9
 * 
 * <p>
 * The operator casts the elements of a given input tensor to a data type
 * specified by the 'to' argument and returns an output tensor of the same size
 * in the converted type. The 'to' argument must be one of the data types
 * specified in the 'DataType' enum field in the TensorProto message.
 * 
 * Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific
 * numeric representations (e.g., "1e-5" and "1E8") to float types is supported.
 * For example, converting string "100.5" to an integer may result 100. There
 * are some string literals reserved for special floating-point values; "+INF"
 * (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and
 * not-a-number, respectively. Any string which can exactly match "+INF" in a
 * case-insensitive way would be mapped to positive infinite. Similarly, this
 * case-insensitive rule is applied to "INF" and "NaN". When casting from
 * numeric tensors to string tensors, plain floating-point representation (such
 * as "314.15926") would be used. Converting non-numerical-literal string such
 * as "Hello World!" is an undefined behavior. Cases of converting string
 * representing floating-point arithmetic value, such as "2.718", to INT is an
 * undefined behavior.
 * 
 * Conversion from a numerical type to any numerical type is always allowed.
 * User must be aware of precision loss and value change caused by range
 * difference between two types. For example, a 64-bit float 3.1415926459 may be
 * round to a 32-bit float 3.141592. Similarly, converting an integer 36 to
 * Boolean may produce 1 because we truncate bits which can't be stored in the
 * targeted type.
 * 
 * @author HarryLee {@literal <formaten@qq.com>}
 * @version 9
 * @since Version 1 of the default ONNX operator set
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Cast-9">ONNX
 *      .Changelog.md</a>
 * @see <a href=
 *      "https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast">ONNX.
 *      Operators.md</a>
 * @see CastV1
 * @see CastV6
 */
public interface CastV9 extends CastV6, AiOnnxOperatorV9 {

	/**
	 * Constrain input types. Casting from complex are not supported.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T1 = new TypeConstraint(DataType.allTypesWithoutComplex());

	/**
	 * Constrain input types. Casting from complex are not supported.
	 */
	public static final TypeConstraint TPYE_CONSTRAINT_T2 = new TypeConstraint(DataType.allTypesWithoutComplex());

	/**
	 * Executes operator
	 * 
	 * @param t1
	 *            Input tensor to be cast.
	 * @param to
	 *            The data type to which the elements of the input tensor are
	 *            cast. Strictly must be one of the types from DataType enum in
	 *            TensorProto
	 * @return Output tensor with the same shape as input with type specified by
	 *         the 'to' argument
	 */
	//public T_TENSOR cast(T_TENSOR t1, Long to);

	class CastInputV9<T_TENSOR> extends CastInputV6<T_TENSOR> {

		private Field<Long> toDTNumberField;

		public CastInputV9(Node node, Inputs inputs) {
			super(node, inputs);

			this.toDTNumberField = new AttributeField<Long>(super.attrs, ATTR_TO, IntAttribute.class, null, true);
		}

		public Long getToDTNumber() {
			return toDTNumberField.getData();
		}

	}

	/**
	 * Outputs for operator execution (forward & backward)
	 *
	 * @param <T_TENSOR>
	 *            The backend tensor object.
	 */
	class CastOutputV9<T_TENSOR> extends CastOutputV6<T_TENSOR> {

		public CastOutputV9(T_TENSOR output) {
			super(output);
		}

		@Override
		public TypeConstraint getTypeConstraint() {
			return TPYE_CONSTRAINT_T2;
		}

	}

}