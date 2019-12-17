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
package org.onnx4j.opsets.aiOnnx.v1;

import java.util.Map;

import org.onnx4j.opsets.Operator;
import org.onnx4j.opsets.aiOnnx.AiOnnxOperatorSetSpec;
import org.onnx4j.opsets.aiOnnx.v1.ops.AbsV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.AddV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ArgMaxV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.AveragePoolV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.BatchNormalizationV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.CastV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ConcatV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ConstantV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ConvV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.DivV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.DropoutV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.IdentityV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ImageScalerV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.LeakyReluV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.MatMulV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.MaxPoolV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.MulV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.PadV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ReluV1;
import org.onnx4j.opsets.aiOnnx.v1.ops.ReshapeV1;

/**
 * Default ONNX Operator Set in version 1
 * 
 * @author HarryLee
 *
 */
public interface AiOnnxOperatorSetSpecV1<T_TENSOR> extends AiOnnxOperatorSetSpec {

	public abstract AbsV1<T_TENSOR> getAbsV1();

	public abstract PadV1<T_TENSOR> getPadV1();

	public abstract MatMulV1<T_TENSOR> getMatMulV1();

	public abstract IdentityV1<T_TENSOR> getIdentityV1();

	public abstract ArgMaxV1<T_TENSOR> getArgMaxV1();

	public abstract DivV1<T_TENSOR> getDivV1();

	public abstract ReshapeV1<T_TENSOR> getReshapeV1();

	public abstract AddV1<T_TENSOR> getAddV1();

	public abstract MaxPoolV1<T_TENSOR> getMaxPoolV1();

	public abstract ReluV1<T_TENSOR> getReluV1();

	public abstract ConvV1<T_TENSOR> getConvV1();

	public abstract ConstantV1<T_TENSOR> getConstantV1();

	public abstract ImageScalerV1<T_TENSOR> getImageScalerV1();

	public abstract BatchNormalizationV1<T_TENSOR> getBatchNormalizationV1();

	public abstract LeakyReluV1<T_TENSOR> getLeakyReluV1();

	public abstract MulV1<T_TENSOR> getMulV1();

	public abstract ConcatV1<T_TENSOR> getConcatV1();

	public abstract DropoutV1<T_TENSOR> getDropoutV1();

	public abstract AveragePoolV1<T_TENSOR> getAveragePoolV1();

	public abstract CastV1<T_TENSOR> getCastV1();

	@Override
	public default Map<String, Operator> initializeOperators() {
		Map<String, Operator> operators = AiOnnxOperatorSetSpec.super.initializeOperators();
		operators.put(AbsV1.OP_TYPE, this.getAbsV1());
		operators.put(PadV1.OP_TYPE, this.getPadV1());
		operators.put(MatMulV1.OP_TYPE, this.getMatMulV1());
		operators.put(IdentityV1.OP_TYPE, this.getIdentityV1());
		operators.put(ArgMaxV1.OP_TYPE, this.getArgMaxV1());
		// 20191024
		operators.put(DivV1.OP_TYPE, this.getDivV1());
		// 20191025
		operators.put(ReshapeV1.OP_TYPE, this.getReshapeV1());
		operators.put(AddV1.OP_TYPE, this.getAddV1());
		operators.put(MaxPoolV1.OP_TYPE, this.getMaxPoolV1());
		// 20191026
		operators.put(ReluV1.OP_TYPE, this.getReluV1());
		operators.put(ConvV1.OP_TYPE, this.getConvV1());
		operators.put(ConstantV1.OP_TYPE, this.getConstantV1());
		// 20191108
		operators.put(LeakyReluV1.OP_TYPE, this.getLeakyReluV1());
		// 20191111
		operators.put(BatchNormalizationV1.OP_TYPE, this.getBatchNormalizationV1());
		operators.put(ImageScalerV1.OP_TYPE, this.getImageScalerV1());
		// 20191115
		operators.put(MulV1.OP_TYPE, this.getMulV1());
		// 20191119
		operators.put(ConcatV1.OP_TYPE, this.getConcatV1());
		// 20191120
		operators.put(DropoutV1.OP_TYPE, this.getDropoutV1());
		operators.put(AveragePoolV1.OP_TYPE, this.getAveragePoolV1());
		// 20191216
		operators.put(CastV1.OP_TYPE, this.getCastV1());
		return operators;
	}

}